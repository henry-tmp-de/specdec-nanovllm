// Paged decode attention —— 手写 CUDA 版（Ampere sm_86 深度实现）
//
// 设计要点（每一条都和 Triton 版对照）
// -----------------------------------
// 1) mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 算 QKᵀ 和 PV。
//    ★ GQA 比只有 4（32 q头 : 8 kv头），mma 的 M 至少 16，所以 M 补到 16、
//      12/16 行喂全 0。这是 Triton 也躲不掉的浪费，本文件同时给出 FFMA M=4
//      的精确版本做对照（paged_decode_ffma_kernel）。
// 2) ldmatrix.sync.aligned.m8n8.x2 把 K/V 从 shared memory 直接搬成 mma fragment。
//    - K 用 **非转置** ldmatrix；V 用 **转置** ldmatrix。推导见 kernel 内注释。
// 3) shared memory 手写 XOR swizzle，消 ldmatrix 的 8 路 bank conflict。见 swz()。
// 4) cp.async.cg 多级流水线（双/三缓冲），把下一块 K/V 的加载压在 mma 下面。
// 5) warp specialization 版本见 paged_decode_ws_kernel。
//
// 布局约定（和 nano-vllm 一致）
//   q           : (num_seqs, num_q_heads, head_dim)                bf16
//   k_cache     : (num_blocks, page_size, num_kv_heads, head_dim)  bf16
//   block_table : (num_seqs, max_blocks)                           int32
//   context_lens: (num_seqs,)                                      int32
//   o           : (num_seqs, num_q_heads, head_dim)                bf16

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <unordered_map>

using bf16 = __nv_bfloat16;
#define DEVINL __device__ __forceinline__
#define FULLMASK 0xffffffffu

namespace pda {

constexpr int HD  = 128;    // head_dim
constexpr int QPK = 4;      // GQA 比：每个 kv 头对应 4 个 q 头
constexpr int MT  = 16;     // mma M（从 QPK=4 补到 16）

// 用 -1e30 而不是 -inf 表示「屏蔽」：exp2(-1e30 - m) 直接得 0，
// 同时避免 inf - inf = NaN。
constexpr float NEG = -1e30f;

// ------------------------------------------------------------------
// shared memory XOR swizzle
//
// tile 是 (rows, HD) 的 bf16，row-major；每行 128 元素 = 256B = 16 个 16B chunk。
//
// 为什么必须做：ldmatrix 一次读 8 行、每行同一列上的 16B。按 row-major 平铺时
// 行距 256B = 64 words，64 mod 32 == 0 —— 8 行的地址落在同一个 bank 上，
// ldmatrix 直接 8 路 conflict，吞吐掉 8 倍。
//
// 做法：行内 16B chunk 索引 c 与行号做 XOR
//     physical_chunk = c ^ (row & 7)
// 同一列 c 第 row 行的 word 地址 = row*64 + ((c ^ (row&7)) * 4)，
// 取 mod 32 后等于 ((c ^ (row&7)) & 7) * 4。row&7 遍历 0..7 时
// (c ^ (row&7)) & 7 是 0..7 的一个排列 -> 8 行落在 8 组互不重叠的 4-word
// bank 组上 -> conflict 归零。
//
// 写侧（cp.async）：同一行相邻 16 个 lane 写 16 个不同 chunk，XOR 只是行内置换，
// 依旧两两不同；跨行也只各占一半 bank —— 512B/128B per cycle = 4 拍，正好是下限。
// ------------------------------------------------------------------
DEVINL uint32_t swz(int row, int cchunk) {
    int pc = cchunk ^ (row & 7);
    return (uint32_t)((row * HD + (pc << 3)) << 1);   // 字节偏移
}

DEVINL uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// ---------------- PTX helpers ----------------
DEVINL void ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}
DEVINL void ldsm_x2_t(uint32_t& r0, uint32_t& r1, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}
DEVINL void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVINL void cp_async16(uint32_t d, const void* s) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(s));
}
DEVINL void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> DEVINL void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

DEVINL uint32_t pack2(float lo, float hi) {
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// ------------------------------------------------------------------
// 把一个 (BLOCK_N, HD) 的 K/V tile 从分页 cache 搬进 shared memory。
// 每线程 KPT 个 16B chunk；chunk 序号 g -> row = g>>4, cc = g&15。
// 相邻 16 个 lane 读同一行的 16 个连续 chunk（256B 连续）-> 完全 coalesced。
// ------------------------------------------------------------------
template<int BLOCK_N, int NT, bool SWIZZLE = true>
DEVINL void issue_tile(const int* __restrict__ bt_row,
                       const bf16* __restrict__ kg,
                       const bf16* __restrict__ vg,
                       bf16* ksm, bf16* vsm,
                       int base_pos, int ctx, int page, int nkvh, int kvh, int tid)
{
    constexpr int CHUNKS = BLOCK_N * (HD / 8);
    constexpr int KPT = CHUNKS / NT;
#pragma unroll
    for (int i = 0; i < KPT; ++i) {
        int g   = tid + i * NT;
        int row = g >> 4;
        int cc  = g & 15;
        int p   = base_pos + row;
        int pc  = p < ctx ? p : ctx - 1;          // 越界位置夹到合法范围（值之后会被 mask）
        int lg  = pc / page;
        int off = pc - lg * page;
        int phys = bt_row[lg];
        if (phys < 0) phys = 0;
        size_t base = ((size_t)phys * page * nkvh + (size_t)off * nkvh + kvh) * HD + cc * 8;
        // SWIZZLE=true  -> ldmatrix 要的 XOR swizzle 布局（mma 版用）
        // SWIZZLE=false -> 朴素 row-major（FFMA 版用：一个 warp 正好铺满一行）
        const uint32_t off_s = SWIZZLE ? swz(row, cc)
                                       : (uint32_t)(((size_t)row * HD + cc * 8) << 1);
        cp_async16(smem_u32(ksm) + off_s, kg + base);
        cp_async16(smem_u32(vsm) + off_s, vg + base);
    }
}

template<int BLOCK_N, int NT, bool SWIZZLE = true>
DEVINL void load_tile_sync(const int* __restrict__ bt_row,
                           const bf16* __restrict__ kg,
                           const bf16* __restrict__ vg,
                           bf16* ksm, bf16* vsm,
                           int base_pos, int ctx, int page, int nkvh, int kvh, int tid)
{
    constexpr int CHUNKS = BLOCK_N * (HD / 8);
    constexpr int KPT = CHUNKS / NT;
#pragma unroll
    for (int i = 0; i < KPT; ++i) {
        int g   = tid + i * NT;
        int row = g >> 4;
        int cc  = g & 15;
        int p   = base_pos + row;
        int pc  = p < ctx ? p : ctx - 1;
        int lg  = pc / page;
        int off = pc - lg * page;
        int phys = bt_row[lg];
        if (phys < 0) phys = 0;
        size_t base = ((size_t)phys * page * nkvh + (size_t)off * nkvh + kvh) * HD + cc * 8;
        uint32_t ko = SWIZZLE ? swz(row, cc)
                              : (uint32_t)(((size_t)row * HD + cc * 8) << 1);
        uint4 kv = *reinterpret_cast<const uint4*>(kg + base);
        uint4 vv = *reinterpret_cast<const uint4*>(vg + base);
        *reinterpret_cast<uint4*>(reinterpret_cast<char*>(ksm) + ko) = kv;
        *reinterpret_cast<uint4*>(reinterpret_cast<char*>(vsm) + ko) = vv;
    }
}

// ------------------------------------------------------------------
// 命名 barrier：warp specialization 的同步原语。
// `bar.arrive` 只登记到达、不阻塞；`bar.sync` 阻塞到 count 个线程都到达。
// count 必须是参与这个 barrier 的**全部**线程数（生产者 + 消费者）。
// ------------------------------------------------------------------
DEVINL void bar_sync_named(int id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}
DEVINL void bar_arrive_named(int id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count) : "memory");
}

// ------------------------------------------------------------------
// 把一块 K/V tile 的 cp.async 发出去（生产者 warp 版：线程编号换成生产者内部的 0..PNT-1）
// ------------------------------------------------------------------
template<int BLOCK_N, int PNT>
DEVINL void issue_tile_p(const int* __restrict__ bt_row,
                         const bf16* __restrict__ kg,
                         const bf16* __restrict__ vg,
                         bf16* ksm, bf16* vsm,
                         int base_pos, int ctx, int page, int nkvh, int kvh, int ptid)
{
    constexpr int CHUNKS = BLOCK_N * (HD / 8);
    constexpr int KPT = CHUNKS / PNT;
    static_assert(CHUNKS % PNT == 0, "chunk 数要能被生产者线程数整除");
#pragma unroll
    for (int i = 0; i < KPT; ++i) {
        int g = ptid + i * PNT;
        int row = g >> 4, cc = g & 15;
        int p = base_pos + row;
        int pc = p < ctx ? p : ctx - 1;
        int lg = pc / page, off = pc - lg * page;
        int phys = bt_row[lg];
        if (phys < 0) phys = 0;
        size_t base = ((size_t)phys * page * nkvh + (size_t)off * nkvh + kvh) * HD + cc * 8;
        cp_async16(smem_u32(ksm) + swz(row, cc), kg + base);
        cp_async16(smem_u32(vsm) + swz(row, cc), vg + base);
    }
}

// ==================================================================
// mma 版的 warp specialization 变体
//
// 布局：CWARPS 个**消费者** warp 负责 mma/softmax，PWARPS 个**生产者** warp 只发 cp.async。
// 同步用命名 barrier 做「stage 就绪 / stage 空闲」两组握手：
//   生产者： 等 free[s] -> 发 cp.async -> commit -> wait_group -> arrive ready[s]
//   消费者： sync ready[s] -> 计算 -> arrive free[s]
// 生产者的循环比消费者长 STAGES-2 轮，形成软件流水（消费者在算 tile t 的时候，
// 生产者在给 tile t+STAGES-1 发拷贝）。
//
// ★ 注意 bar.sync 的 count 必须是 (CWARPS+PWARPS)*32 —— 两边都算在内。
// ==================================================================
template<int BLOCK_N, int CWARPS, int PWARPS, int STAGES>
__global__ __launch_bounds__((CWARPS + PWARPS) * 32)
void paged_decode_ws_kernel(
    const bf16* __restrict__ qg,
    const bf16* __restrict__ kg,
    const bf16* __restrict__ vg,
    const int*  __restrict__ bt,
    const int*  __restrict__ cl,
    float*      __restrict__ pm,
    float*      __restrict__ pl,
    float*      __restrict__ pacc,
    float scale2,
    int num_q_heads, int num_kv_heads, int page, int max_blocks,
    int chunk, int splits)
{
    constexpr int CTOTAL = (CWARPS + PWARPS) * 32;
    constexpr int PNT = PWARPS * 32;
    constexpr int PW = BLOCK_N / CWARPS;
    static_assert(PW % 16 == 0, "PW 必须 ≥16 且是 16 的倍数");
    constexpr int NKT = HD / 16, NNQ = PW / 8, NNP = HD / 8, PKT = PW / 16;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* q_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* k_sm = q_sm + 8 * HD;
    bf16* v_sm = k_sm + (size_t)STAGES * BLOCK_N * HD;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kvh = blockIdx.x, split = blockIdx.y, seq = blockIdx.z;
    const int ctx = cl[seq];
    const int start = split * chunk;
    const int valid_end = min(start + chunk, ctx);
    const int ntiles = (valid_end > start) ? ((valid_end - start + BLOCK_N - 1) / BLOCK_N) : 0;
    const int* bt_row = bt + (size_t)seq * max_blocks;

    for (int i = tid; i < 8 * HD; i += CTOTAL) {
        int r = i / HD, c = i % HD;
        bf16 v = __float2bfloat16(0.f);
        if (r < QPK) v = qg[((size_t)seq * num_q_heads + kvh * QPK + r) * HD + c];
        q_sm[r * HD + ((c >> 3) ^ (r & 7)) * 8 + (c & 7)] = v;
    }
    __syncthreads();          // ★ 必须所有线程都到，所以放在 if/else 之外

    // barrier id: ready[s] = 1+s, free[s] = 1+STAGES+s
    const int ready0 = 1, free0 = 1 + STAGES;

    // 消费者累加器提到 if/else 外面声明：循环后的跨 warp 合并要用到它们
    float acc[NNP][4];
    float mrow = NEG, lrow = 0.f;
#pragma unroll
    for (int i = 0; i < NNP; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
    const int wpos = start + warp * PW;

    if (warp >= CWARPS) {
        // ---------------- 生产者 ----------------
        // 关键：`cp.async.wait_group K` 只保证「最近 K 个 group 之前」的拷贝完成。
        // 流水线里每个 tile 提交一个 group，所以信号 tile tr 时用 wait_group<STAGES-2>
        // 正好让 group tr 落地；**但尾巴那 STAGES-2 轮没有真实拷贝**，置空 group 的
        // 计数在硬件上不保证，所以尾巴里改用一个保守的 wait_all。
        // （不这么做会出现「大部分元素对、少数元素错」的静默算错，因为消费者会
        //   在数据没落地之前就被放行。）
        const int ptid = tid - CWARPS * 32;
        for (int t = 0; t < ntiles + STAGES - 2; ++t) {
            const bool live = (t < ntiles);
            if (live) {
                const int s = t % STAGES;
                if (t >= STAGES) bar_sync_named(free0 + s, CTOTAL);
                issue_tile_p<BLOCK_N, PNT>(bt_row, kg, vg,
                                           k_sm + (size_t)s * BLOCK_N * HD,
                                           v_sm + (size_t)s * BLOCK_N * HD,
                                           start + t * BLOCK_N, ctx, page,
                                           num_kv_heads, kvh, ptid);
            }
            cp_commit();
            const int tr = t - (STAGES - 2);
            if (tr >= 0) {
                if (live) cp_wait<STAGES - 2>();
                else      cp_wait<0>();
                __threadfence_block();        // 让 smem 的写对其它 warp 可见
                bar_arrive_named(ready0 + (tr % STAGES), CTOTAL);
            }
        }
        cp_wait<0>();
    } else {
        // ---------------- 消费者 ----------------
        uint32_t aq[NKT][4];
#pragma unroll
        for (int kt = 0; kt < NKT; ++kt) {
            int r = lane & 15, row = (r & 7), ck = 2 * kt + (r >> 3);
            uint32_t x0, x1;
            ldsm_x2(x0, x1, smem_u32(q_sm) + swz(row, ck));
            aq[kt][0] = x0; aq[kt][1] = 0u; aq[kt][2] = x1; aq[kt][3] = 0u;
        }

        for (int t = 0; t < ntiles; ++t) {
            const int s = t % STAGES;
            bf16* ks = k_sm + (size_t)s * BLOCK_N * HD;
            bf16* vs = v_sm + (size_t)s * BLOCK_N * HD;
            bar_sync_named(ready0 + s, CTOTAL);
            __threadfence_block();

            float c[NNQ][4];
#pragma unroll
            for (int nt = 0; nt < NNQ; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
#pragma unroll
            for (int kt = 0; kt < NKT; ++kt) {
                uint32_t a[4] = {aq[kt][0], aq[kt][1], aq[kt][2], aq[kt][3]};
#pragma unroll
                for (int nt = 0; nt < NNQ; ++nt) {
                    int r = lane & 15;
                    int row = warp * PW + nt * 8 + (r & 7);
                    int ck = 2 * kt + (r >> 3);
                    uint32_t b0, b1;
                    ldsm_x2(b0, b1, smem_u32(ks) + swz(row, ck));
                    mma16816(c[nt], a, b0, b1);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NNQ; ++nt) {
                int col = wpos + nt * 8 + (lane & 3) * 2;
                c[nt][0] = (col     < valid_end) ? c[nt][0] * scale2 : NEG;
                c[nt][1] = (col + 1 < valid_end) ? c[nt][1] * scale2 : NEG;
                c[nt][2] = 0.f; c[nt][3] = 0.f;
            }
            float mx = c[0][0];
#pragma unroll
            for (int nt = 0; nt < NNQ; ++nt) { mx = fmaxf(mx, c[nt][0]); mx = fmaxf(mx, c[nt][1]); }
            mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 2));
            float m_new = fmaxf(mrow, mx);
            float alpha = exp2f(mrow - m_new);
            float p[NNQ][2], lsum = 0.f;
#pragma unroll
            for (int nt = 0; nt < NNQ; ++nt) {
                p[nt][0] = (c[nt][0] <= NEG * 0.5f) ? 0.f : exp2f(c[nt][0] - m_new);
                p[nt][1] = (c[nt][1] <= NEG * 0.5f) ? 0.f : exp2f(c[nt][1] - m_new);
                lsum += p[nt][0] + p[nt][1];
            }
            lsum += __shfl_xor_sync(FULLMASK, lsum, 1);
            lsum += __shfl_xor_sync(FULLMASK, lsum, 2);
            lrow = lrow * alpha + lsum;
            mrow = m_new;
#pragma unroll
            for (int i = 0; i < NNP; ++i) { acc[i][0] *= alpha; acc[i][1] *= alpha; }

#pragma unroll
            for (int kk = 0; kk < PKT; ++kk) {
                uint32_t ap[4];
                ap[0] = pack2(p[2 * kk][0], p[2 * kk][1]);
                ap[1] = 0u;
                ap[2] = pack2(p[2 * kk + 1][0], p[2 * kk + 1][1]);
                ap[3] = 0u;
#pragma unroll
                for (int nt = 0; nt < NNP; ++nt) {
                    int row = warp * PW + kk * 16 + (lane & 15);
                    uint32_t b0, b1;
                    ldsm_x2_t(b0, b1, smem_u32(vs) + swz(row, nt));
                    mma16816(acc[nt], ap, b0, b1);
                }
            }
            bar_arrive_named(free0 + s, CTOTAL);
        }

    }

    // ★ 消费者不能在这里直接写 k_sm：生产者 warp 的循环比消费者长，
    //   可能还在往 k_sm 的 stage 里发 cp.async。必须等所有线程都退出循环
    //   （也就是生产者把最后一块也发完）再复用这块 smem，否则就是静默算错。
    __syncthreads();
    if (warp < CWARPS) {
        float* red_a = reinterpret_cast<float*>(k_sm);
        float* red_m = red_a + (size_t)CWARPS * QPK * HD;
        float* red_l = red_m + CWARPS * QPK;
        if (lane < 16) {  // noqa
            int h = lane >> 2;
            red_m[warp * 16 + h] = mrow;
            red_l[warp * 16 + h] = lrow;
#pragma unroll
            for (int nt = 0; nt < NNP; ++nt) {
                red_a[(warp * QPK + h) * HD + nt * 8 + (lane & 3) * 2]     = acc[nt][0];
                red_a[(warp * QPK + h) * HD + nt * 8 + (lane & 3) * 2 + 1] = acc[nt][1];
            }
        }
    }
    __syncthreads();
    if (warp < CWARPS) {
        float* red_a = reinterpret_cast<float*>(k_sm);
        float* red_m = red_a + (size_t)CWARPS * QPK * HD;
        float* red_l = red_m + CWARPS * QPK;
        size_t hbase = ((size_t)seq * num_kv_heads + kvh) * splits + split;
        for (int i = tid; i < QPK * HD; i += CWARPS * 32) {
            int h = i / HD, d = i - h * HD;
            float M = NEG, L = 0.f, A = 0.f;
#pragma unroll
            for (int w = 0; w < CWARPS; ++w) M = fmaxf(M, red_m[w * 16 + h]);
#pragma unroll
            for (int w = 0; w < CWARPS; ++w) {
                float wt = exp2f(red_m[w * 16 + h] - M);
                L += red_l[w * 16 + h] * wt;
                A += red_a[(w * QPK + h) * HD + d] * wt;
            }
            pm[hbase * QPK + h] = M;
            pl[hbase * QPK + h] = L;
            pacc[(hbase * QPK + h) * HD + d] = A;
        }
    }
}

// ==================================================================
// FFMA 版：不用 tensor core，正好算 M = QPK = 4 行
//
// 为什么写它：核心假设是「Triton 的 tl.dot 要求 M≥16，GQA 比只有 4，所以
// 75% 的 MMA 行是白算的；CUDA 可以正好算 4 行」。这个 kernel 就是那个
// 「正好 4 行」的实现，用来和 mma 版对拍。
//
// 布局上的关键区别：mma 版必须把 K/V 摆成 ldmatrix 要的 16B-chunk swizzle；
// FFMA 版反而是**朴素 row-major 最好**——
//   lane l 读 k[row][4l .. 4l+4]（8 字节），一个 warp 的 32 个 lane 正好铺满
//   一整行 128 个 bf16 = 256B => 一次 LDS.64 就是 256B、无 bank conflict。
//   所以这个 kernel 里**一个 swizzle 都不需要**，这本身就是个好对照。
//
// 代价：head_dim 方向的归约要靠 warp shuffle。每个 (位置, q头) 要做
// 5 步 butterfly，所以 shuffle 数量成为它相对 mma 版的主要劣势 —— 这是
// 实测出来的结论，不是先验判断（见 README 的阶梯表）。
// ==================================================================
template<int BLOCK_N, int WARPS, int STAGES, int NB>
__global__ __launch_bounds__(WARPS * 32)
void paged_decode_ffma_kernel(
    const bf16* __restrict__ qg,
    const bf16* __restrict__ kg,
    const bf16* __restrict__ vg,
    const int*  __restrict__ bt,
    const int*  __restrict__ cl,
    float*      __restrict__ pm,
    float*      __restrict__ pl,
    float*      __restrict__ pacc,
    float scale2,
    int num_q_heads, int num_kv_heads, int page, int max_blocks,
    int chunk, int splits)
{
    constexpr int NT = WARPS * 32;
    constexpr int PW = BLOCK_N / WARPS;
    static_assert(PW % NB == 0, "PW 必须是 NB 的倍数");
    static_assert((BLOCK_N * (HD / 8)) % NT == 0, "chunk/thread 必须整除");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* q_sm = reinterpret_cast<bf16*>(smem_raw);            // QPK x HD（朴素布局）
    bf16* k_sm = q_sm + QPK * HD;                              // STAGES x BLOCK_N x HD
    bf16* v_sm = k_sm + (size_t)STAGES * BLOCK_N * HD;
    // 跨 warp 合并的暂存直接叠在 k_sm 上（循环结束后 K stage 已经不用了），
    // 省 8KB smem —— sm_86 每个 block 最多只能用 99KB，省下来这点是 3 级流水线
    // 能不能塞下的关键。

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kvh = blockIdx.x, split = blockIdx.y, seq = blockIdx.z;
    const int ctx = cl[seq];
    const int start = split * chunk;
    const int valid_end = min(start + chunk, ctx);
    const int ntiles = (valid_end > start) ? ((valid_end - start + BLOCK_N - 1) / BLOCK_N) : 0;
    const int* bt_row = bt + (size_t)seq * max_blocks;

    // Q 进 smem（朴素布局，只要 QPK 行）
    for (int i = tid; i < QPK * HD; i += NT) {
        int h = i / HD, d = i - h * HD;
        q_sm[i] = qg[((size_t)seq * num_q_heads + kvh * QPK + h) * HD + d];
    }
    __syncthreads();

    // lane l 常驻保存 q[h][4l .. 4l+4]（4 个头 × 4 个 dim = 8 个 uint32）
    uint32_t qreg[QPK][2];
#pragma unroll
    for (int h = 0; h < QPK; ++h) {
        uint2 v = *reinterpret_cast<const uint2*>(q_sm + h * HD + lane * 4);
        qreg[h][0] = v.x; qreg[h][1] = v.y;
    }

    float acc[QPK][4] = {};
    float mrow[QPK], lrow[QPK];
#pragma unroll
    for (int h = 0; h < QPK; ++h) { mrow[h] = NEG; lrow[h] = 0.f; }

    if (STAGES > 1) {
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < ntiles)
                issue_tile<BLOCK_N, NT, false>(bt_row, kg, vg,
                                               k_sm + (size_t)s * BLOCK_N * HD,
                                               v_sm + (size_t)s * BLOCK_N * HD,
                                               start + s * BLOCK_N, ctx, page,
                                               num_kv_heads, kvh, tid);
            cp_commit();
        }
    }

    for (int t = 0; t < ntiles; ++t) {
        const int s = (STAGES > 1) ? (t % STAGES) : 0;
        bf16* ks = k_sm + (size_t)s * BLOCK_N * HD;
        bf16* vs = v_sm + (size_t)s * BLOCK_N * HD;
        if (STAGES > 1) {
            cp_wait<STAGES - 2>();
            __syncthreads();
            if (t + STAGES - 1 < ntiles) {
                int s2 = (t + STAGES - 1) % STAGES;
                issue_tile<BLOCK_N, NT, false>(bt_row, kg, vg,
                                               k_sm + (size_t)s2 * BLOCK_N * HD,
                                               v_sm + (size_t)s2 * BLOCK_N * HD,
                                               start + (t + STAGES - 1) * BLOCK_N,
                                               ctx, page, num_kv_heads, kvh, tid);
            }
            cp_commit();
        } else {
            __syncthreads();
            load_tile_sync<BLOCK_N, NT, false>(bt_row, kg, vg, ks, vs,
                                               start + t * BLOCK_N, ctx, page,
                                               num_kv_heads, kvh, tid);
            __syncthreads();
        }

#pragma unroll
        for (int b0 = 0; b0 < PW; b0 += NB) {
            // ---- 1) QKᵀ：每个 lane 先算自己 4 个 dim 的部分积，再 butterfly ----
            float qk[NB][QPK];
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                int j = warp * PW + b0 + i;
                uint2 k2 = *reinterpret_cast<const uint2*>(ks + j * HD + lane * 4);
                __nv_bfloat162 ka = *reinterpret_cast<__nv_bfloat162*>(&k2.x);
                __nv_bfloat162 kb = *reinterpret_cast<__nv_bfloat162*>(&k2.y);
#pragma unroll
                for (int h = 0; h < QPK; ++h) {
                    __nv_bfloat162 qa = *reinterpret_cast<__nv_bfloat162*>(&qreg[h][0]);
                    __nv_bfloat162 qb = *reinterpret_cast<__nv_bfloat162*>(&qreg[h][1]);
                    float part = __bfloat162float(qa.x) * __bfloat162float(ka.x)
                               + __bfloat162float(qa.y) * __bfloat162float(ka.y)
                               + __bfloat162float(qb.x) * __bfloat162float(kb.x)
                               + __bfloat162float(qb.y) * __bfloat162float(kb.y);
                    // 5 步 butterfly：32 个 lane 合起来正好是 128 个 dim 的全和
                    part += __shfl_xor_sync(FULLMASK, part, 1);
                    part += __shfl_xor_sync(FULLMASK, part, 2);
                    part += __shfl_xor_sync(FULLMASK, part, 4);
                    part += __shfl_xor_sync(FULLMASK, part, 8);
                    part += __shfl_xor_sync(FULLMASK, part, 16);
                    qk[i][h] = part;
                }
            }

            // ---- 2) 屏蔽 + 缩放 ----
            int colbase = start + t * BLOCK_N + warp * PW + b0;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                bool ok = (colbase + i) < valid_end;
#pragma unroll
                for (int h = 0; h < QPK; ++h) qk[i][h] = ok ? qk[i][h] * scale2 : NEG;
            }

            // ---- 3) V 先读进寄存器（batch 内只读一次，供 4 个头复用）----
            float vf[NB][4];
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                int j = warp * PW + b0 + i;
                uint2 v2 = *reinterpret_cast<const uint2*>(vs + j * HD + lane * 4);
                __nv_bfloat162 va = *reinterpret_cast<__nv_bfloat162*>(&v2.x);
                __nv_bfloat162 vb = *reinterpret_cast<__nv_bfloat162*>(&v2.y);
                vf[i][0] = __bfloat162float(va.x);
                vf[i][1] = __bfloat162float(va.y);
                vf[i][2] = __bfloat162float(vb.x);
                vf[i][3] = __bfloat162float(vb.y);
            }

            // ---- 4) online softmax + PV（m/l 在 butterfly 后全 lane 一致）----
#pragma unroll
            for (int h = 0; h < QPK; ++h) {
                float mx = mrow[h];
#pragma unroll
                for (int i = 0; i < NB; ++i) mx = fmaxf(mx, qk[i][h]);
                float alpha = exp2f(mrow[h] - mx);
                float pv[NB], lsum = 0.f;
#pragma unroll
                for (int i = 0; i < NB; ++i) {
                    pv[i] = (qk[i][h] <= NEG * 0.5f) ? 0.f : exp2f(qk[i][h] - mx);
                    lsum += pv[i];
                }
                lrow[h] = lrow[h] * alpha + lsum;
                mrow[h] = mx;
#pragma unroll
                for (int d = 0; d < 4; ++d) acc[h][d] *= alpha;
#pragma unroll
                for (int i = 0; i < NB; ++i) {
#pragma unroll
                    for (int d = 0; d < 4; ++d) acc[h][d] += pv[i] * vf[i][d];
                }
            }
        }
    }

    if (STAGES > 1) cp_wait<0>();
    __syncthreads();

    // ---- 跨 warp 合并：acc[h][4l..4l+4] 直接落 smem，再按 (h,d) 求和 ----
    // red 布局：[WARPS][QPK][HD] float + [WARPS][QPK] m + [WARPS][QPK] l
    float* red_a = reinterpret_cast<float*>(k_sm);
    float* red_m = red_a + (size_t)WARPS * QPK * HD;
    float* red_l = red_m + WARPS * QPK;
#pragma unroll
    for (int h = 0; h < QPK; ++h) {
#pragma unroll
        for (int d = 0; d < 4; ++d) red_a[(warp * QPK + h) * HD + lane * 4 + d] = acc[h][d];
        if (lane == 0) { red_m[warp * QPK + h] = mrow[h]; red_l[warp * QPK + h] = lrow[h]; }
    }
    __syncthreads();

    size_t hbase = ((size_t)seq * num_kv_heads + kvh) * splits + split;
    for (int i = tid; i < QPK * HD; i += NT) {
        int h = i / HD, d = i - h * HD;
        float M = NEG, L = 0.f, A = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) M = fmaxf(M, red_m[w * QPK + h]);
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            float wt = exp2f(red_m[w * QPK + h] - M);
            L += red_l[w * QPK + h] * wt;
            A += red_a[(w * QPK + h) * HD + d] * wt;
        }
        pm[hbase * QPK + h] = M;
        pl[hbase * QPK + h] = L;
        pacc[(hbase * QPK + h) * HD + d] = A;
    }
}

// ------------------------------------------------------------------
// 纯读取探针：访存部分和主 kernel **逐字节一致**（同一套 issue_tile、同一个
// cp.async 流水线、同一个 grid），但把 mma/softmax 全部删掉。
//
// 用途：把「访存地板」和「计算/同步开销」分开。如果这个 kernel 的时间和主 kernel
// 差不多，说明主 kernel 已经贴着访存地板了，再优化计算没用；
// 如果明显更快，说明差距在计算侧。
// ------------------------------------------------------------------
template<int BLOCK_N, int WARPS, int STAGES>
__global__ __launch_bounds__(WARPS * 32)
void paged_decode_readonly_kernel(
    const bf16* __restrict__ kg,
    const bf16* __restrict__ vg,
    const int*  __restrict__ bt,
    const int*  __restrict__ cl,
    float*      __restrict__ sink,
    int nkvh, int page, int max_blocks, int chunk)
{
    constexpr int NT = WARPS * 32;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* k_sm = reinterpret_cast<bf16*>(smem_raw);
    bf16* v_sm = k_sm + (size_t)STAGES * BLOCK_N * HD;

    const int tid = threadIdx.x;
    const int kvh = blockIdx.x, split = blockIdx.y, seq = blockIdx.z;
    const int ctx = cl[seq];
    const int start = split * chunk;
    const int valid_end = min(start + chunk, ctx);
    const int ntiles = (valid_end > start) ? ((valid_end - start + BLOCK_N - 1) / BLOCK_N) : 0;
    const int* bt_row = bt + (size_t)seq * max_blocks;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < ntiles)
            issue_tile<BLOCK_N, NT>(bt_row, kg, vg,
                                    k_sm + (size_t)s * BLOCK_N * HD,
                                    v_sm + (size_t)s * BLOCK_N * HD,
                                    start + s * BLOCK_N, ctx, page, nkvh, kvh, tid);
        cp_commit();
    }
    for (int t = 0; t < ntiles; ++t) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (t + STAGES - 1 < ntiles) {
            int s2 = (t + STAGES - 1) % STAGES;
            issue_tile<BLOCK_N, NT>(bt_row, kg, vg,
                                    k_sm + (size_t)s2 * BLOCK_N * HD,
                                    v_sm + (size_t)s2 * BLOCK_N * HD,
                                    start + (t + STAGES - 1) * BLOCK_N,
                                    ctx, page, nkvh, kvh, tid);
        }
        cp_commit();
    }
    cp_wait<0>();
    __syncthreads();
    // 永不成立的条件，纯粹为了别让编译器认为 smem 没用（cp.async 是 volatile asm，
    // 其实删不掉，这里只是加一道保险）
    if (k_sm[tid] == __float2bfloat16(-12345.0f) &&
        v_sm[tid] == __float2bfloat16(-54321.0f)) sink[0] = 1.0f;
}

// ------------------------------------------------------------------
// split 之间的归约（融合进主 kernel 的版本）
//
// 每个 CTA 把自己的 (M, L, A) 写到显存后，__threadfence() 再 atomicAdd 一个计数器；
// **最后一个到达的 CTA** 顺手把这一组的 splits 份 partial 归约掉并写输出。
// 这是 CUDA 经典的 threadfence-reduction（CUB / flash-decoding single-pass 同款），
// 好处是**完全省掉第二个 kernel 的启动**。
//
// ★ 为什么这里要手动做 4 路展开：splits 是运行期变量，编译器不会展开，
//   于是 `M = fmax(M, pm[s])` 会串成一条 [load -> max -> load -> max] 的
//   依赖链，每个 split 一次显存往返。第一版的独立 combine kernel 就是这么
//   慢到 8us 的。手动 4 路展开后一次能飞出去 4 个 load。
// ------------------------------------------------------------------
DEVINL void fused_reduce(int seq, int kvh, int splits, int num_kv_heads,
                         int num_q_heads,
                         const float* __restrict__ pm,
                         const float* __restrict__ pl,
                         const float* __restrict__ pacc,
                         bf16* __restrict__ og,
                         int tid, int warp, int lane)
{
    if (warp >= QPK) return;               // 正好 QPK 个 warp 各管一个 q 头
    const int h = warp;
    const size_t base = ((size_t)seq * num_kv_heads + kvh) * splits;
    const float* pmh = pm + base * QPK + h;
    const float* plh = pl + base * QPK + h;

    float M = NEG;
    int s = 0;
    for (; s + 4 <= splits; s += 4) {
        float t0 = pmh[(s + 0) * QPK], t1 = pmh[(s + 1) * QPK];
        float t2 = pmh[(s + 2) * QPK], t3 = pmh[(s + 3) * QPK];
        M = fmaxf(M, fmaxf(fmaxf(t0, t1), fmaxf(t2, t3)));
    }
    for (; s < splits; ++s) M = fmaxf(M, pmh[s * QPK]);

    float L = 0.f;
    s = 0;
    for (; s + 4 <= splits; s += 4) {
        float t0 = plh[(s + 0) * QPK] * exp2f(pmh[(s + 0) * QPK] - M);
        float t1 = plh[(s + 1) * QPK] * exp2f(pmh[(s + 1) * QPK] - M);
        float t2 = plh[(s + 2) * QPK] * exp2f(pmh[(s + 2) * QPK] - M);
        float t3 = plh[(s + 3) * QPK] * exp2f(pmh[(s + 3) * QPK] - M);
        L += (t0 + t1) + (t2 + t3);
    }
    for (; s < splits; ++s) L += plh[s * QPK] * exp2f(pmh[s * QPK] - M);
    const float inv = (L > 0.f) ? (1.f / L) : 1.f;

    const float* pa = pacc + (base * QPK + h) * HD + lane * 4;
    float4 ac = make_float4(0.f, 0.f, 0.f, 0.f);
    s = 0;
    for (; s + 4 <= splits; s += 4) {
        const float4 v0 = *reinterpret_cast<const float4*>(pa + (size_t)(s + 0) * QPK * HD);
        const float4 v1 = *reinterpret_cast<const float4*>(pa + (size_t)(s + 1) * QPK * HD);
        const float4 v2 = *reinterpret_cast<const float4*>(pa + (size_t)(s + 2) * QPK * HD);
        const float4 v3 = *reinterpret_cast<const float4*>(pa + (size_t)(s + 3) * QPK * HD);
        const float w0 = exp2f(pmh[(s + 0) * QPK] - M), w1 = exp2f(pmh[(s + 1) * QPK] - M);
        const float w2 = exp2f(pmh[(s + 2) * QPK] - M), w3 = exp2f(pmh[(s + 3) * QPK] - M);
        ac.x += v0.x * w0 + v1.x * w1 + v2.x * w2 + v3.x * w3;
        ac.y += v0.y * w0 + v1.y * w1 + v2.y * w2 + v3.y * w3;
        ac.z += v0.z * w0 + v1.z * w1 + v2.z * w2 + v3.z * w3;
        ac.w += v0.w * w0 + v1.w * w1 + v2.w * w2 + v3.w * w3;
    }
    for (; s < splits; ++s) {
        const float4 v = *reinterpret_cast<const float4*>(pa + (size_t)s * QPK * HD);
        const float w = exp2f(pmh[s * QPK] - M);
        ac.x += v.x * w; ac.y += v.y * w; ac.z += v.z * w; ac.w += v.w * w;
    }

    bf16* o = og + ((size_t)seq * num_q_heads + kvh * QPK + h) * HD + lane * 4;
    o[0] = __float2bfloat16(ac.x * inv);
    o[1] = __float2bfloat16(ac.y * inv);
    o[2] = __float2bfloat16(ac.z * inv);
    o[3] = __float2bfloat16(ac.w * inv);
    (void)tid;
}

// ------------------------------------------------------------------
// 主 kernel：mma + ldmatrix + XOR swizzle（+ 可选 cp.async 流水线）
//
// grid  = (num_kv_heads, num_splits, num_seqs)   block = WARPS*32
// 每个 CTA 管一个 (序列, kv头, split)。队内 WARPS 个 warp 再沿位置方向切开，
// 各自维护 (m, l, acc)，循环结束后在 shared memory 里跨 warp 合并 ——
// 这样 split 内部不需要第二个 kernel（Triton v3 的 partial 之间是分开的）。
// ------------------------------------------------------------------
template<int BLOCK_N, int WARPS, int STAGES, bool CPASYNC>
__global__ __launch_bounds__(WARPS * 32)
void paged_decode_mma_kernel(
    const bf16* __restrict__ qg,
    const bf16* __restrict__ kg,
    const bf16* __restrict__ vg,
    const int*  __restrict__ bt,
    const int*  __restrict__ cl,
    float*      __restrict__ pm,     // (nseq, nkvh, splits, QPK)
    float*      __restrict__ pl,
    float*      __restrict__ pacc,   // (nseq, nkvh, splits, QPK, HD)
    bf16*       __restrict__ og,     // 只有 fused 模式才用（最后一个 CTA 直接写输出）
    int*        __restrict__ ctr,    // splits 计数器；nullptr = 不做 kernel 内归约
    float scale2,                    // softmax_scale * log2(e)
    int num_q_heads, int num_kv_heads, int page, int max_blocks,
    int chunk, int splits)
{
    constexpr int NT  = WARPS * 32;
    constexpr int PW  = BLOCK_N / WARPS;
    static_assert(PW % 16 == 0, "PW must be a multiple of 16");
    constexpr int NKT = HD / 16;             // QK 的 k 方向 tile 数（head_dim）= 8
    constexpr int NNQ = PW / 8;              // QK 的 n 方向 tile 数（位置）
    constexpr int NNP = HD / 8;              // PV 的 n 方向 tile 数（head_dim）= 16
    constexpr int PKT = PW / 16;             // PV 的 k 方向 tile 数（位置）
    static_assert((BLOCK_N * (HD / 8)) % NT == 0, "chunk/thread 必须整除");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* q_sm = reinterpret_cast<bf16*>(smem_raw);              // 8 x HD（行 4..7 补 0）
    bf16* k_sm = q_sm + 8 * HD;                                  // STAGES x BLOCK_N x HD
    bf16* v_sm = k_sm + (size_t)STAGES * BLOCK_N * HD;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kvh  = blockIdx.x;
    const int split= blockIdx.y;
    const int seq  = blockIdx.z;

    const int ctx       = cl[seq];
    const int start     = split * chunk;
    const int valid_end = min(start + chunk, ctx);
    const int ntiles    = (valid_end > start) ? ((valid_end - start + BLOCK_N - 1) / BLOCK_N) : 0;

    const int* bt_row = bt + (size_t)seq * max_blocks;

    // ---- Q 搬进 shared memory：行 0..3 是真 q，行 4..7 补 0 ----
    // 为什么只需要 8 行：mma 的 A fragment 里 a0a1 取 M 行 L>>2（0..7），
    // a2a3 / a6a7 取 M 行 8..15 —— 那 8 行是补 0 行，直接用常量 0，
    // 根本不读 smem，所以只需要行 0..7。
    for (int i = tid; i < 8 * HD; i += NT) {
        int r = i / HD, c = i % HD;
        bf16 v = __float2bfloat16(0.f);
        if (r < QPK)
            v = qg[((size_t)seq * num_q_heads + kvh * QPK + r) * HD + c];
        q_sm[r * HD + ((c >> 3) ^ (r & 7)) * 8 + (c & 7)] = v;
    }

    // ---- Q 的 A fragment（只读一次，常驻寄存器）----
    // 非转置 ldmatrix.x2：lane0..7 给 matrix0 的 8 个行地址，lane8..15 给 matrix1 的。
    //   matrix0 行 i 取 q_sm[i][k0]   -> 结果 lane 拿到 M0[L>>2][(L&3)*2,+1]
    //   matrix1 行 i 取 q_sm[i][k0+8] -> 结果 lane 拿到 M1[L>>2][(L&3)*2,+1]
    // 恰好就是 a0a1（列 k0..k0+8）和 a4a5（列 k0+8..k0+16）。
    // ★ 必须同步：q_sm 是别的线程写进去的，下面 ldmatrix 要读它。
    //   少这一句就是一个真 race —— 症状是「大部分 ctx 都对，个别 ctx 差 1e-2」，
    //   非常像精度问题，其实是读到没写完的 shared memory。
    __syncthreads();

    uint32_t aq[NKT][4];
#pragma unroll
    for (int kt = 0; kt < NKT; ++kt) {
        int r   = lane & 15;
        int row = (r & 7);
        int ck  = 2 * kt + (r >> 3);
        uint32_t x0, x1;
        ldsm_x2(x0, x1, smem_u32(q_sm) + swz(row, ck));
        aq[kt][0] = x0; aq[kt][1] = 0u; aq[kt][2] = x1; aq[kt][3] = 0u;
    }

    float acc[NNP][4];
#pragma unroll
    for (int i = 0; i < NNP; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
    float mrow = NEG, lrow = 0.f;

    const int wpos = start + warp * PW;

    // ---- cp.async 预热 ----
    if (CPASYNC) {
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < ntiles)
                issue_tile<BLOCK_N, NT>(bt_row, kg, vg,
                                        k_sm + (size_t)s * BLOCK_N * HD,
                                        v_sm + (size_t)s * BLOCK_N * HD,
                                        start + s * BLOCK_N, ctx, page, num_kv_heads, kvh, tid);
            cp_commit();
        }
    }

    for (int t = 0; t < ntiles; ++t) {
        const int s = CPASYNC ? (t % STAGES) : 0;
        bf16* ks = k_sm + (size_t)s * BLOCK_N * HD;
        bf16* vs = v_sm + (size_t)s * BLOCK_N * HD;

        if (CPASYNC) {
            cp_wait<STAGES - 2>();
            __syncthreads();
            if (t + STAGES - 1 < ntiles) {
                int s2 = (t + STAGES - 1) % STAGES;
                issue_tile<BLOCK_N, NT>(bt_row, kg, vg,
                                        k_sm + (size_t)s2 * BLOCK_N * HD,
                                        v_sm + (size_t)s2 * BLOCK_N * HD,
                                        start + (t + STAGES - 1) * BLOCK_N,
                                        ctx, page, num_kv_heads, kvh, tid);
            }
            cp_commit();
        } else {
            __syncthreads();
            load_tile_sync<BLOCK_N, NT>(bt_row, kg, vg, ks, vs,
                                        start + t * BLOCK_N, ctx, page, num_kv_heads, kvh, tid);
            __syncthreads();
        }

        // ================= QKᵀ =================
        // C = Q · Kᵀ，C[m][n] = Σ_d Q[m][d] K[n][d]（m 是 q 头，n 是位置）。
        // mma 的 A 是 (M,K)=(16,16) row-major，B 是 (K,N)=(16,8) col-major。
        // 这里 B[k][n] = Kᵀ[k][n] = K[n][k]，要求 fragment
        //     b0b1 = B[k=(L&3)*2,+1][n=L>>2] = K[pos=L>>2][dim=(L&3)*2,+1]
        // 非转置 ldmatrix 的结果是 M[L>>2][(L&3)*2,+1]，行地址由 lane0..7 给出。
        // 令 matrix0 的行 i 指向 &k_sm[nbase+i][d0]，立刻得到
        //     K[nbase+L>>2][d0+(L&3)*2,+1]  ✓ 正好是 b0b1
        // matrix1 的行 i 指向 &k_sm[nbase+i][d0+8] -> b2b3 ✓
        float c[NNQ][4];
#pragma unroll
        for (int nt = 0; nt < NNQ; ++nt) { c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f; }

#pragma unroll
        for (int kt = 0; kt < NKT; ++kt) {
            uint32_t a[4] = {aq[kt][0], aq[kt][1], aq[kt][2], aq[kt][3]};
#pragma unroll
            for (int nt = 0; nt < NNQ; ++nt) {
                int r   = lane & 15;
                // ★ K tile 是整个 CTA 共用的，每个 warp 只取自己那 PW 个位置的行
                int row = warp * PW + nt * 8 + (r & 7);
                int ck  = 2 * kt + (r >> 3);
                uint32_t b0, b1;
                ldsm_x2(b0, b1, smem_u32(ks) + swz(row, ck));
                mma16816(c[nt], a, b0, b1);
            }
        }

        // ---- 越界位置屏蔽 + 缩放 ----
#pragma unroll
        for (int nt = 0; nt < NNQ; ++nt) {
            int col = wpos + nt * 8 + (lane & 3) * 2;
            c[nt][0] = (col     < valid_end) ? c[nt][0] * scale2 : NEG;
            c[nt][1] = (col + 1 < valid_end) ? c[nt][1] * scale2 : NEG;
            c[nt][2] = 0.f;    // M 行 8..15：A 行恒 0，结果恒 0，直接丢弃
            c[nt][3] = 0.f;
        }

        // ================= online softmax =================
        float mx = c[0][0];
#pragma unroll
        for (int nt = 0; nt < NNQ; ++nt) {
            mx = fmaxf(mx, c[nt][0]);
            mx = fmaxf(mx, c[nt][1]);
        }
        // 同组的 4 个 lane（L>>2 相同）合起来覆盖整行的列 -> 2 步 butterfly
        mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(FULLMASK, mx, 2));

        float m_new = fmaxf(mrow, mx);
        float alpha = exp2f(mrow - m_new);

        float p[NNQ][2];
        float lsum = 0.f;
#pragma unroll
        for (int nt = 0; nt < NNQ; ++nt) {
            p[nt][0] = (c[nt][0] <= NEG * 0.5f) ? 0.f : exp2f(c[nt][0] - m_new);
            p[nt][1] = (c[nt][1] <= NEG * 0.5f) ? 0.f : exp2f(c[nt][1] - m_new);
            lsum += p[nt][0] + p[nt][1];
        }
        lsum += __shfl_xor_sync(FULLMASK, lsum, 1);
        lsum += __shfl_xor_sync(FULLMASK, lsum, 2);
        lrow = lrow * alpha + lsum;
        mrow = m_new;

        // acc 重缩放：只有 M 行 0..7（acc[.][0],[1]）需要；行 8..15 恒 0
#pragma unroll
        for (int i = 0; i < NNP; ++i) { acc[i][0] *= alpha; acc[i][1] *= alpha; }

        // ================= PV =================
        // C = P · V，C[m][n] = Σ_k P[m][k] V[k][n]（k 是位置，n 是 head_dim）。
        // B[k][n] = V[k][n]，要 b0b1 = B[k=(L&3)*2,+1][n=L>>2] = V[pos=(L&3)*2,+1][dim=L>>2]。
        // 转置 ldmatrix 的结果是 M[(L&3)*2,+1][L>>2]（转置后的 M[L>>2][...]）。
        // 令 matrix0 的行 i 指向 &v_sm[nbase+i][d0]，得到
        //     V[nbase+(L&3)*2,+1][d0+L>>2]  ✓ 正好是 b0b1
        // matrix1 的行 i 指向 &v_sm[nbase+8+i][d0] -> b2b3 ✓
        // 两个合起来就是「行 = nbase + (lane&15)」这一个公式。
#pragma unroll
        for (int kk = 0; kk < PKT; ++kk) {
            uint32_t ap[4];
            ap[0] = pack2(p[2 * kk][0], p[2 * kk][1]);          // A 行 L>>2，列 0..8
            ap[1] = 0u;                                          // A 行 8..15 -> 0
            ap[2] = pack2(p[2 * kk + 1][0], p[2 * kk + 1][1]);  // A 行 L>>2，列 8..16
            ap[3] = 0u;
#pragma unroll
            for (int nt = 0; nt < NNP; ++nt) {
                // ★ 同上：V tile 也是 CTA 共用的，warp 只读自己那段位置的行
                int row = warp * PW + kk * 16 + (lane & 15);
                uint32_t b0, b1;
                ldsm_x2_t(b0, b1, smem_u32(vs) + swz(row, nt));
                mma16816(acc[nt], ap, b0, b1);
            }
        }
    }

    if (CPASYNC) { cp_wait<0>(); }
    __syncthreads();

    // ================= CTA 内跨 warp 合并 =================
    // K/V 的 stage 缓冲此时已经用完，直接叠在上面当暂存（省 8KB+ smem）
    float* red_m = reinterpret_cast<float*>(k_sm);
    float* red_l = red_m + WARPS * 16;
    float* red_a = red_l + WARPS * 16;

    if (lane < 16) {
        int h = lane >> 2;
        red_m[warp * 16 + h] = mrow;
        red_l[warp * 16 + h] = lrow;
#pragma unroll
        for (int nt = 0; nt < NNP; ++nt) {
            red_a[(warp * QPK + h) * HD + nt * 8 + (lane & 3) * 2]     = acc[nt][0];
            red_a[(warp * QPK + h) * HD + nt * 8 + (lane & 3) * 2 + 1] = acc[nt][1];
        }
    }
    __syncthreads();

    size_t hbase = ((size_t)seq * num_kv_heads + kvh) * splits + split;
    for (int i = tid; i < QPK * HD; i += NT) {
        int h = i / HD, d = i - h * HD;
        float M = NEG, L = 0.f, A = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) M = fmaxf(M, red_m[w * 16 + h]);
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            float wt = exp2f(red_m[w * 16 + h] - M);
            L += red_l[w * 16 + h] * wt;
            A += red_a[(w * QPK + h) * HD + d] * wt;
        }
        pm[hbase * QPK + h] = M;
        pl[hbase * QPK + h] = L;
        pacc[(hbase * QPK + h) * HD + d] = A;
    }

    // ---- 融合归约：最后一个到达的 CTA 顺手把这一组做完 ----
    if (ctr != nullptr) {
        __threadfence();                 // 自己的 partial 先对所有 CTA 可见
        __syncthreads();
        __shared__ int s_last;
        if (tid == 0) {
            int old = atomicAdd(&ctr[seq * num_kv_heads + kvh], 1);
            s_last = (old == splits - 1) ? 1 : 0;
        }
        __syncthreads();
        if (s_last) {
            fused_reduce(seq, kvh, splits, num_kv_heads, num_q_heads,
                         pm, pl, pacc, og, tid, warp, lane);
            __syncthreads();
            if (tid == 0) ctr[seq * num_kv_heads + kvh] = 0;   // 复位给下一次用
        }
    }
}

// ------------------------------------------------------------------
// split 之间的归约。
//
// ★ 第一版这里慢到 8.0us（比主 kernel 的 1/3 还多），原因是三处
//   `for (s = 0; s < splits; ++s) acc = f(acc, load(...))` 的**串行依赖链**：
//   splits 是运行期变量，编译器不能展开，于是一条 load -> 一条 max/add -> 下一条 load
//   全串起来，16 个 split 就是 16 次显存往返 ≈ 16 × 600ns ≈ 9.6us。
//   小 kernel 上「延迟」比「带宽」重要得多。
//
//   改法两条：
//     (1) 把 SPLITS 做成模板参数 -> #pragma unroll 全展开 -> 16 个 load 同时飞出去；
//     (2) 每个 lane 用 float4 读 4 个连续 dim，一个 warp 一次读满 512B 连续，
//         避免「每线程跨 2KB 步长读 16 个标量」造成的 sector 放大。
// ------------------------------------------------------------------
template<int SPLITS>
__global__ void paged_decode_combine_kernel(
    const float* __restrict__ pm,
    const float* __restrict__ pl,
    const float* __restrict__ pacc,
    bf16*        __restrict__ og,
    int num_q_heads, int num_kv_heads)
{
    const int seq  = blockIdx.y;
    const int kvh  = blockIdx.x;
    const int h    = threadIdx.x >> 5;      // warp -> 一个 q 头（正好 QPK 个 warp）
    const int lane = threadIdx.x & 31;
    const size_t base = ((size_t)seq * num_kv_heads + kvh) * SPLITS;

    const float* pmh = pm + base * QPK + h;
    const float* plh = pl + base * QPK + h;

    float M = NEG;
#pragma unroll
    for (int s = 0; s < SPLITS; ++s) M = fmaxf(M, pmh[s * QPK]);

    float wts[SPLITS];
    float L = 0.f;
#pragma unroll
    for (int s = 0; s < SPLITS; ++s) {
        wts[s] = exp2f(pmh[s * QPK] - M);
        L += plh[s * QPK] * wts[s];
    }
    const float inv = (L > 0.f) ? (1.f / L) : 1.f;

    const float* pa = pacc + ((base * QPK) + h) * HD + lane * 4;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < SPLITS; ++s) {
        float4 v = *reinterpret_cast<const float4*>(pa + (size_t)s * QPK * HD);
        a.x += v.x * wts[s];
        a.y += v.y * wts[s];
        a.z += v.z * wts[s];
        a.w += v.w * wts[s];
    }
    bf16* o = og + ((size_t)seq * num_q_heads + kvh * QPK + h) * HD + lane * 4;
    o[0] = __float2bfloat16(a.x * inv);
    o[1] = __float2bfloat16(a.y * inv);
    o[2] = __float2bfloat16(a.z * inv);
    o[3] = __float2bfloat16(a.w * inv);
}

// dim 方向只有 HD/lane/4 = 1 个 lane 组，所以每线程就 4 个 dim；上面的实现覆盖 HD=128。
static_assert(HD % (32 * 4) == 0, "combine kernel 假设每 lane 4 个 dim");

// ---------------- warp specialization 版 launcher ----------------
template<int BLOCK_N, int CW, int PW_, int STAGES>
size_t smem_ws() {
    return (size_t)(8 * HD + 2 * STAGES * BLOCK_N * HD) * sizeof(bf16) + 64;
}

template<int BLOCK_N, int CW, int PW_, int STAGES>
void launch_ws_impl(const bf16* q, const bf16* k, const bf16* v,
                    const int* bt, const int* cl,
                    float* pm, float* pl, float* pacc, float scale2,
                    int nseq, int nqh, int nkvh, int page, int max_blocks,
                    int chunk, int splits, cudaStream_t st)
{
    auto* kern = pda::paged_decode_ws_kernel<BLOCK_N, CW, PW_, STAGES>;
    size_t smem = smem_ws<BLOCK_N, CW, PW_, STAGES>();
    static bool done = false;
    if (!done) {
        cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        TORCH_CHECK(e == cudaSuccess, "WS smem ", smem, " 超上限: ", cudaGetErrorString(e));
        done = true;
    }
    dim3 grid(nkvh, splits, nseq);
    kern<<<grid, (CW + PW_) * 32, smem, st>>>(q, k, v, bt, cl, pm, pl, pacc, scale2,
                                              nqh, nkvh, page, max_blocks, chunk, splits);
}

template<int BLOCK_N, int CW, int PW_>
void dispatch_ws_stages(const bf16* q, const bf16* k, const bf16* v,
                        const int* bt, const int* cl,
                        float* pm, float* pl, float* pacc, float scale2,
                        int nseq, int nqh, int nkvh, int page, int max_blocks,
                        int chunk, int splits, int stages, cudaStream_t st)
{
    switch (stages) {
        case 2: launch_ws_impl<BLOCK_N, CW, PW_, 2>(q,k,v,bt,cl,pm,pl,pacc,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        case 3: launch_ws_impl<BLOCK_N, CW, PW_, 3>(q,k,v,bt,cl,pm,pl,pacc,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        case 4: launch_ws_impl<BLOCK_N, CW, PW_, 4>(q,k,v,bt,cl,pm,pl,pacc,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        default: TORCH_CHECK(false, "WS: stages 只支持 2..4");
    }
}

// ---------------- FFMA 版 launcher ----------------
template<int BLOCK_N, int WARPS, int STAGES, int NB>
size_t smem_ffma() {
    return (size_t)(QPK * HD + 2 * STAGES * BLOCK_N * HD) * sizeof(bf16) + 64;
}

template<int BLOCK_N, int WARPS, int STAGES, int NB>
void launch_ffma_impl(const bf16* q, const bf16* k, const bf16* v,
                      const int* bt, const int* cl,
                      float* pm, float* pl, float* pacc, float scale2,
                      int nseq, int nqh, int nkvh, int page, int max_blocks,
                      int chunk, int splits, cudaStream_t st)
{
    auto* kern = pda::paged_decode_ffma_kernel<BLOCK_N, WARPS, STAGES, NB>;
    size_t smem = smem_ffma<BLOCK_N, WARPS, STAGES, NB>();
    static bool done = false;
    if (!done) {
        cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        TORCH_CHECK(e == cudaSuccess, "FFMA smem ", smem, " 超过上限: ", cudaGetErrorString(e));
        done = true;
    }
    dim3 grid(nkvh, splits, nseq);
    kern<<<grid, WARPS * 32, smem, st>>>(q, k, v, bt, cl, pm, pl, pacc, scale2,
                                         nqh, nkvh, page, max_blocks, chunk, splits);
}

template<int BLOCK_N, int WARPS, int STAGES>
void dispatch_ffma(const bf16* q, const bf16* k, const bf16* v,
                   const int* bt, const int* cl,
                   float* pm, float* pl, float* pacc, float scale2,
                   int nseq, int nqh, int nkvh, int page, int max_blocks,
                   int chunk, int splits, cudaStream_t st)
{
    static_assert((BLOCK_N / WARPS) % 8 == 0, "FFMA: PW 必须是 8 的倍数");
    launch_ffma_impl<BLOCK_N, WARPS, STAGES, 8>(q,k,v,bt,cl,pm,pl,pacc,scale2,
                                                nseq,nqh,nkvh,page,max_blocks,
                                                chunk,splits,st);
}

template<int BLOCK_N, int WARPS>
void dispatch_ffma_stages(const bf16* q, const bf16* k, const bf16* v,
                          const int* bt, const int* cl,
                          float* pm, float* pl, float* pacc, float scale2,
                          int nseq, int nqh, int nkvh, int page, int max_blocks,
                          int chunk, int splits, int stages, cudaStream_t st)
{
    switch (stages) {
        case 1: dispatch_ffma<BLOCK_N, WARPS, 1>(q,k,v,bt,cl,pm,pl,pacc,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        case 2: dispatch_ffma<BLOCK_N, WARPS, 2>(q,k,v,bt,cl,pm,pl,pacc,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        case 3: dispatch_ffma<BLOCK_N, WARPS, 3>(q,k,v,bt,cl,pm,pl,pacc,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        case 4: dispatch_ffma<BLOCK_N, WARPS, 4>(q,k,v,bt,cl,pm,pl,pacc,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        default: TORCH_CHECK(false, "stages 只支持 1..4");
    }
}

template<int BLOCK_N, int WARPS, int STAGES>
void launch_readonly_impl(const bf16* k, const bf16* v, const int* bt,
                          const int* cl, float* sink, int nseq, int nkvh,
                          int page, int max_blocks, int chunk, int splits,
                          cudaStream_t st)
{
    size_t smem = (size_t)(2 * STAGES * BLOCK_N * HD) * sizeof(bf16) + 64;
    auto* kern = pda::paged_decode_readonly_kernel<BLOCK_N, WARPS, STAGES>;
    static bool done = false;
    if (!done) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        done = true;
    }
    dim3 grid(nkvh, splits, nseq);
    kern<<<grid, WARPS * 32, smem, st>>>(k, v, bt, cl, sink, nkvh, page,
                                         max_blocks, chunk);
}

torch::Tensor read_only(torch::Tensor k_cache, torch::Tensor v_cache,
                        torch::Tensor block_table, torch::Tensor context_lens,
                        int64_t splits, int64_t block_n, int64_t warps,
                        int64_t stages)
{
    const at::cuda::OptionalCUDAGuard guard(k_cache.device());
    auto kc = k_cache.contiguous(), vc = v_cache.contiguous();
    auto btc = block_table.contiguous(), clc = context_lens.contiguous();
    const int nseq = clc.size(0), nkvh = kc.size(2), page = kc.size(1);
    const int max_blocks = btc.size(1);
    auto sink = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32)
                                                 .device(kc.device()));
    int max_ctx = max_blocks * page;
    long chunk = ((max_ctx + splits - 1) / splits + block_n - 1) / block_n * block_n;
    if (chunk < block_n) chunk = block_n;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const bf16* kp = reinterpret_cast<const bf16*>(kc.data_ptr());
    const bf16* vp = reinterpret_cast<const bf16*>(vc.data_ptr());
    const int* bp = btc.data_ptr<int>();
    const int* cp = clc.data_ptr<int>();
    float* sp = sink.data_ptr<float>();
#define RO_CASE(BN, W, ST) launch_readonly_impl<BN, W, ST>(kp,vp,bp,cp,sp,nseq,nkvh,page,max_blocks,(int)chunk,(int)splits,st)
    if (block_n == 64 && warps == 4) {
        if      (stages == 2) RO_CASE(64,4,2);
        else if (stages == 3) RO_CASE(64,4,3);
        else if (stages == 4) RO_CASE(64,4,4);
        else TORCH_CHECK(false, "read_only: 不支持的 stages");
    } else if (block_n == 32 && warps == 2) {
        if      (stages == 2) RO_CASE(32,2,2);
        else if (stages == 3) RO_CASE(32,2,3);
        else if (stages == 4) RO_CASE(32,2,4);
        else TORCH_CHECK(false, "read_only: 不支持的 stages");
    } else {
        TORCH_CHECK(false, "read_only: 只支持 (64,4) / (32,2) 组合");
    }
#undef RO_CASE
    return sink;
}

}  // namespace pda

// ==================================================================
// host side
// ==================================================================
namespace {

using pda::HD; using pda::QPK;

template<int BLOCK_N, int WARPS, int STAGES, bool CPASYNC>
size_t smem_for() {
    return (size_t)(8 * HD + 2 * STAGES * BLOCK_N * HD) * sizeof(bf16)
         + 64 /* 对齐余量 */;
}

template<int BLOCK_N, int WARPS, int STAGES, bool CPASYNC>
void launch_one(const bf16* q, const bf16* k, const bf16* v,
                const int* bt, const int* cl,
                float* pm, float* pl, float* pacc,
                bf16* og, int* ctr,
                float scale2, int nseq, int nqh, int nkvh, int page,
                int max_blocks, int chunk, int splits, cudaStream_t st)
{
    auto* kern = pda::paged_decode_mma_kernel<BLOCK_N, WARPS, STAGES, CPASYNC>;
    size_t smem = smem_for<BLOCK_N, WARPS, STAGES, CPASYNC>();
    static bool done = false;
    if (!done) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        done = true;
    }
    dim3 grid(nkvh, splits, nseq);
    kern<<<grid, WARPS * 32, smem, st>>>(q, k, v, bt, cl, pm, pl, pacc, og, ctr,
                                         scale2, nqh, nkvh, page, max_blocks,
                                         chunk, splits);
}

template<int BLOCK_N, int WARPS>
void dispatch_stages(const bf16* q, const bf16* k, const bf16* v,
                     const int* bt, const int* cl,
                     float* pm, float* pl, float* pacc,
                     bf16* og, int* ctr,
                     float scale2, int nseq, int nqh, int nkvh, int page,
                     int max_blocks, int chunk, int splits, int stages, cudaStream_t st)
{
    switch (stages) {
        case 1: launch_one<BLOCK_N, WARPS, 1, false>(q,k,v,bt,cl,pm,pl,pacc,og,ctr,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        case 2: launch_one<BLOCK_N, WARPS, 2, true >(q,k,v,bt,cl,pm,pl,pacc,og,ctr,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        case 3: launch_one<BLOCK_N, WARPS, 3, true >(q,k,v,bt,cl,pm,pl,pacc,og,ctr,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        case 4: launch_one<BLOCK_N, WARPS, 4, true >(q,k,v,bt,cl,pm,pl,pacc,og,ctr,scale2,nseq,nqh,nkvh,page,max_blocks,chunk,splits,st); break;
        default: TORCH_CHECK(false, "stages 只支持 1..4");
    }
}

struct LaunchArgs {
    const bf16 *q, *k, *v; const int *bt, *cl;
    float *pm, *pl, *pacc; bf16* og; int* ctr;
    float scale2;
    int nseq, nqh, nkvh, page, max_blocks, chunk, splits, stages;
    cudaStream_t st;
};

template<int BLOCK_N, int WARPS>
void do_launch(const LaunchArgs& a) {
    dispatch_stages<BLOCK_N, WARPS>(a.q, a.k, a.v, a.bt, a.cl, a.pm, a.pl, a.pacc,
                                    a.og, a.ctr, a.scale2, a.nseq, a.nqh, a.nkvh,
                                    a.page, a.max_blocks, a.chunk, a.splits,
                                    a.stages, a.st);
}

// 计数器缓冲：一次性分配 + 清零，之后每次 kernel 自己复位，不用再 memset。
// key = (device, nseq*nkvh)
torch::Tensor& counter_for(int nseq, int nkvh, const torch::TensorOptions& opts) {
    static std::unordered_map<long, torch::Tensor> cache;
    long key = (long)nseq * 1000003L + nkvh;
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, torch::zeros({nseq, nkvh},
                    opts.dtype(torch::kInt32))).first;
    }
    return it->second;
}

void launch_combine(const torch::Tensor& pm, const torch::Tensor& pl,
                    const torch::Tensor& pacc, torch::Tensor& og,
                    int nqh, int nkvh, int splits, cudaStream_t st)
{
    dim3 grid(nkvh, pm.size(0));
    const float* pmp = pm.data_ptr<float>();
    const float* plp = pl.data_ptr<float>();
    const float* pap = pacc.data_ptr<float>();
    bf16* op = reinterpret_cast<bf16*>(og.data_ptr());
    switch (splits) {
        case 1:  pda::paged_decode_combine_kernel<1 ><<<grid, QPK*32, 0, st>>>(pmp,plp,pap,op,nqh,nkvh); break;
        case 2:  pda::paged_decode_combine_kernel<2 ><<<grid, QPK*32, 0, st>>>(pmp,plp,pap,op,nqh,nkvh); break;
        case 4:  pda::paged_decode_combine_kernel<4 ><<<grid, QPK*32, 0, st>>>(pmp,plp,pap,op,nqh,nkvh); break;
        case 8:  pda::paged_decode_combine_kernel<8 ><<<grid, QPK*32, 0, st>>>(pmp,plp,pap,op,nqh,nkvh); break;
        case 16: pda::paged_decode_combine_kernel<16><<<grid, QPK*32, 0, st>>>(pmp,plp,pap,op,nqh,nkvh); break;
        case 32: pda::paged_decode_combine_kernel<32><<<grid, QPK*32, 0, st>>>(pmp,plp,pap,op,nqh,nkvh); break;
        case 64: pda::paged_decode_combine_kernel<64><<<grid, QPK*32, 0, st>>>(pmp,plp,pap,op,nqh,nkvh); break;
        default: TORCH_CHECK(false, "splits 只支持 1/2/4/8/16/32/64");
    }
}

}  // namespace

torch::Tensor paged_decode_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t splits, int64_t block_n, int64_t warps, int64_t stages,
    bool fused)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    auto qc  = q.contiguous();
    auto kc  = k_cache.contiguous();
    auto vc  = v_cache.contiguous();
    auto btc = block_table.contiguous();
    auto clc = context_lens.contiguous();

    const int nseq = qc.size(0);
    const int nqh  = qc.size(1);
    const int nkvh = kc.size(2);
    const int page = kc.size(1);
    const int max_blocks = btc.size(1);
    TORCH_CHECK(nqh == nkvh * QPK, "只支持 GQA 比 4");
    TORCH_CHECK(qc.size(2) == HD, "只支持 head_dim 128");

    auto o = torch::empty_like(qc);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(qc.device());
    auto pm   = torch::empty({nseq, nkvh, splits, QPK}, opts);
    auto pl   = torch::empty_like(pm);
    auto pacc = torch::empty({nseq, nkvh, splits, QPK, HD}, opts);

    const float scale2 = (float)(scale * 1.4426950408889634);   // * log2(e)

    int max_ctx = max_blocks * page;
    long chunk = ((max_ctx + splits - 1) / splits + block_n - 1) / block_n * block_n;
    if (chunk < block_n) chunk = block_n;

    cudaStream_t st = at::cuda::getCurrentCUDAStream();

    int* ctr = nullptr;
    if (fused) {
        ctr = counter_for(nseq, nkvh, opts).data_ptr<int>();
    }

    LaunchArgs a{reinterpret_cast<const bf16*>(qc.data_ptr()),
                 reinterpret_cast<const bf16*>(kc.data_ptr()),
                 reinterpret_cast<const bf16*>(vc.data_ptr()),
                 btc.data_ptr<int>(), clc.data_ptr<int>(),
                 pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(),
                 reinterpret_cast<bf16*>(o.data_ptr()), ctr,
                 scale2, nseq, nqh, nkvh, page, max_blocks, (int)chunk,
                 (int)splits, (int)stages, st};

    if      (block_n == 32  && warps == 2) do_launch<32, 2>(a);
    else if (block_n == 64  && warps == 2) do_launch<64, 2>(a);
    else if (block_n == 64  && warps == 4) do_launch<64, 4>(a);
    else if (block_n == 128 && warps == 4) do_launch<128, 4>(a);
    else if (block_n == 128 && warps == 8) do_launch<128, 8>(a);
    else TORCH_CHECK(false, "不支持的 (block_n, warps) 组合");

    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "main kernel launch failed: ", cudaGetErrorString(e));

    if (!fused) {
        launch_combine(pm, pl, pacc, o, nqh, nkvh, (int)splits, st);
        e = cudaGetLastError();
        TORCH_CHECK(e == cudaSuccess, "combine kernel launch failed: ", cudaGetErrorString(e));
    }
    return o;
}

// ============ warp specialization 版 ============
torch::Tensor paged_decode_ws_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t splits, int64_t block_n, int64_t cwarps,
    int64_t pwarps, int64_t stages)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    auto qc  = q.contiguous();   auto kc  = k_cache.contiguous();
    auto vc  = v_cache.contiguous();
    auto btc = block_table.contiguous(); auto clc = context_lens.contiguous();
    const int nseq = qc.size(0), nqh = qc.size(1), nkvh = kc.size(2);
    const int page = kc.size(1), max_blocks = btc.size(1);
    TORCH_CHECK(nqh == nkvh * QPK && qc.size(2) == HD, "只支持 GQA=4 / head_dim=128");

    auto o = torch::empty_like(qc);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(qc.device());
    auto pm   = torch::empty({nseq, nkvh, splits, QPK}, opts);
    auto pl   = torch::empty_like(pm);
    auto pacc = torch::empty({nseq, nkvh, splits, QPK, HD}, opts);
    const float scale2 = (float)(scale * 1.4426950408889634);
    int max_ctx = max_blocks * page;
    long chunk = ((max_ctx + splits - 1) / splits + block_n - 1) / block_n * block_n;
    if (chunk < block_n) chunk = block_n;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const bf16* qp = reinterpret_cast<const bf16*>(qc.data_ptr());
    const bf16* kp = reinterpret_cast<const bf16*>(kc.data_ptr());
    const bf16* vp = reinterpret_cast<const bf16*>(vc.data_ptr());
    const int*  bp = btc.data_ptr<int>();  const int* cp = clc.data_ptr<int>();
    float* pmp = pm.data_ptr<float>(); float* plp = pl.data_ptr<float>();
    float* pap = pacc.data_ptr<float>();

    // 目前只开 (64, 4消费+4生产) 这一组：PW=16，正好是 mma 需要的最小 n 方向粒度
    if      (block_n == 64 && cwarps == 4 && pwarps == 4)
        pda::dispatch_ws_stages<64, 4, 4>(qp,kp,vp,bp,cp,pmp,plp,pap,scale2,nseq,nqh,nkvh,page,max_blocks,(int)chunk,(int)splits,(int)stages,st);
    else if (block_n == 64 && cwarps == 4 && pwarps == 2)
        pda::dispatch_ws_stages<64, 4, 2>(qp,kp,vp,bp,cp,pmp,plp,pap,scale2,nseq,nqh,nkvh,page,max_blocks,(int)chunk,(int)splits,(int)stages,st);
    else TORCH_CHECK(false, "WS: 只支持 (64, 消费4/生产4) 或 (64, 消费4/生产2)");

    launch_combine(pm, pl, pacc, o, nqh, nkvh, (int)splits, st);
    return o;
}

// ============ FFMA M=4 版（不用 tensor core）============
torch::Tensor paged_decode_ffma_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t splits, int64_t block_n, int64_t warps, int64_t stages)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    auto qc  = q.contiguous();
    auto kc  = k_cache.contiguous();
    auto vc  = v_cache.contiguous();
    auto btc = block_table.contiguous();
    auto clc = context_lens.contiguous();

    const int nseq = qc.size(0), nqh = qc.size(1), nkvh = kc.size(2);
    const int page = kc.size(1), max_blocks = btc.size(1);
    TORCH_CHECK(nqh == nkvh * QPK && qc.size(2) == HD, "只支持 GQA=4 / head_dim=128");

    auto o = torch::empty_like(qc);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(qc.device());
    auto pm   = torch::empty({nseq, nkvh, splits, QPK}, opts);
    auto pl   = torch::empty_like(pm);
    auto pacc = torch::empty({nseq, nkvh, splits, QPK, HD}, opts);
    const float scale2 = (float)(scale * 1.4426950408889634);

    int max_ctx = max_blocks * page;
    long chunk = ((max_ctx + splits - 1) / splits + block_n - 1) / block_n * block_n;
    if (chunk < block_n) chunk = block_n;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const bf16* qp = reinterpret_cast<const bf16*>(qc.data_ptr());
    const bf16* kp = reinterpret_cast<const bf16*>(kc.data_ptr());
    const bf16* vp = reinterpret_cast<const bf16*>(vc.data_ptr());
    const int*  bp = btc.data_ptr<int>();
    const int*  cp = clc.data_ptr<int>();
    float* pmp = pm.data_ptr<float>();
    float* plp = pl.data_ptr<float>();
    float* pap = pacc.data_ptr<float>();

    if      (block_n == 64 && warps == 4) pda::dispatch_ffma_stages<64, 4>(qp,kp,vp,bp,cp,pmp,plp,pap,scale2,nseq,nqh,nkvh,page,max_blocks,(int)chunk,(int)splits,(int)stages,st);
    else if (block_n == 64 && warps == 2) pda::dispatch_ffma_stages<64, 2>(qp,kp,vp,bp,cp,pmp,plp,pap,scale2,nseq,nqh,nkvh,page,max_blocks,(int)chunk,(int)splits,(int)stages,st);
    else TORCH_CHECK(false, "FFMA 只支持 (64,4) / (64,2)");

    launch_combine(pm, pl, pacc, o, nqh, nkvh, (int)splits, st);
    return o;
}

// 只跑 combine kernel（把两段时间拆开量的时候用）
torch::Tensor paged_decode_cuda_combine(
    torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t splits)
{
    const at::cuda::OptionalCUDAGuard guard(pm.device());
    auto o = torch::empty({pm.size(0), num_q_heads, pda::HD},
                          torch::TensorOptions().dtype(torch::kBFloat16)
                                                .device(pm.device()));
    launch_combine(pm, pl, pacc, o, (int)num_q_heads, (int)num_kv_heads,
                   (int)splits, at::cuda::getCurrentCUDAStream());
    return o;
}

// 只跑主 kernel，不做 combine —— 用来把两段时间拆开看
torch::Tensor paged_decode_cuda_partial(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t splits, int64_t block_n, int64_t warps, int64_t stages)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    auto qc = q.contiguous(); auto kc = k_cache.contiguous();
    auto vc = v_cache.contiguous(); auto btc = block_table.contiguous();
    auto clc = context_lens.contiguous();
    const int nseq = qc.size(0), nqh = qc.size(1), nkvh = kc.size(2),
              page = kc.size(1), max_blocks = btc.size(1);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(qc.device());
    auto pm   = torch::empty({nseq, nkvh, splits, QPK}, opts);
    auto pl   = torch::empty_like(pm);
    auto pacc = torch::empty({nseq, nkvh, splits, QPK, HD}, opts);
    const float scale2 = (float)(scale * 1.4426950408889634);
    int max_ctx = max_blocks * page;
    long chunk = ((max_ctx + splits - 1) / splits + block_n - 1) / block_n * block_n;
    if (chunk < block_n) chunk = block_n;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    LaunchArgs a{reinterpret_cast<const bf16*>(qc.data_ptr()),
                 reinterpret_cast<const bf16*>(kc.data_ptr()),
                 reinterpret_cast<const bf16*>(vc.data_ptr()),
                 btc.data_ptr<int>(), clc.data_ptr<int>(),
                 pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(),
                 nullptr, nullptr,
                 scale2, nseq, nqh, nkvh, page, max_blocks, (int)chunk,
                 (int)splits, (int)stages, st};
    if      (block_n == 32  && warps == 2) do_launch<32, 2>(a);
    else if (block_n == 64  && warps == 2) do_launch<64, 2>(a);
    else if (block_n == 64  && warps == 4) do_launch<64, 4>(a);
    else if (block_n == 128 && warps == 4) do_launch<128, 4>(a);
    else if (block_n == 128 && warps == 8) do_launch<128, 8>(a);
    else TORCH_CHECK(false, "不支持的 (block_n, warps) 组合");
    return pm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_decode", &paged_decode_cuda, "paged decode attention (CUDA)");
    m.def("partial", &paged_decode_cuda_partial, "main kernel only");
    m.def("combine_only", &paged_decode_cuda_combine, "combine kernel only");
    m.def("read_only", &pda::read_only, "read-only probe with identical memory access");
    m.def("ffma", &paged_decode_ffma_cuda, "FFMA M=4 (no tensor core) variant");
    m.def("ws", &paged_decode_ws_cuda, "warp-specialized variant");
}
