import sys
sys.path.insert(0, "/home/ziru/nano-vllm/p1-work")
from nanovllm.engine.block_manager import BlockManager


class StubSeq:
    def __init__(self, n, bs):
        self.n = n
        self.bs = bs
        self.block_table = list(range((n + bs - 1) // bs))
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = n

    def block(self, i):
        return [0] * self.bs


def crashes(n, bs=256):
    nblocks = (n + bs - 1) // bs
    bm = BlockManager(max(nblocks + 4, 64), bs)
    s = StubSeq(n, bs)
    s.num_cached_tokens += s.num_scheduled_tokens   # postprocess 里的顺序
    try:
        bm.hash_blocks(s)
        return False
    except IndexError:
        return True


def ranges(xs):
    out = []
    for x in xs:
        if out and x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return out


for n in (250, 255, 256, 300, 500, 511, 512, 513, 700, 768, 1000, 1023, 1024, 2048, 4096, 8192):
    print("prompt=%5d tokens -> %s" % (n, "CRASH(IndexError)" if crashes(n) else "ok"))

bad = [n for n in range(1, 4200) if crashes(n)]
print()
print("crash 的最小 n:", min(bad) if bad else None)
print("崩溃 n 的区间(前10个):", ranges(bad)[:10])
