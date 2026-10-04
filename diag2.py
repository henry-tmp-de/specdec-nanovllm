"""逐步打印：提议了什么、logits 是什么、接受了几个。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B")
llm = LLM(MODEL, enforce_eager=True, max_num_batched_tokens=16384,
          spec_k=3, spec_batch_threshold=0)
sp = SamplingParams(temperature=1e-4, max_tokens=10, ignore_eos=True)
llm.add_request("The capital of France is", sp)

step = 0
while not llm.is_finished() and step < 5:
    seqs, is_pf = llm.scheduler.schedule()
    if is_pf:
        print(f"\n=== step {step}: PREFILL (n={seqs[0].num_scheduled_tokens}) ===")
        llm.model_runner.call("run", seqs, True)
        llm.scheduler.postprocess(seqs, [15043], True)  # 随便给个 token 号让它跑下去
        step += 1
        continue

    s = seqs[0]
    n = s.num_scheduled_tokens
    print(f"\n=== step {step}: DECODE  n_scheduled={n}  num_tokens={len(s)} ===")
    print(f"  token_ids 末尾: {s.token_ids[-6:]}")
    print(f"  draft_tokens : {s.draft_tokens}")

    # 手动跑一遍，看 logits
    llm.model_runner.spec_proposer = llm.model_runner.spec_proposer
    from nanovllm.spec_decode.verify import verify_batch
    for sq in seqs:
        sq.draft_tokens = llm.model_runner.spec_proposer.propose(sq.token_ids, 3)
    input_ids, positions = llm.model_runner.prepare_verify(seqs)
    print(f"  送进 forward 的 input_ids : {input_ids.tolist()}")
    print(f"  送进 forward 的 positions : {positions.tolist()}")
    logits = llm.model_runner.run_model(input_ids, positions, True)
    from nanovllm.utils.context import reset_context
    reset_context()

    cu = llm.model_runner._last_cu_seqlens_q
    print(f"  cu_seqlens_q: {cu}")
    sub = logits[cu[0]:cu[0]+n].unsqueeze(0)
    probs = torch.softmax(sub[0].float(), dim=-1)
    for j in range(n):
        top = probs[j].argmax().item()
        print(f"    位置{j}(pos={positions[j].item()}): argmax={top}  p={probs[j, top]:.4f}")
    if s.draft_tokens:
        for j, d in enumerate(s.draft_tokens):
            print(f"    draft[{j}]={d} 在位置{j+1}的 p={probs[j+1, d]:.6f}")

    toks = llm.model_runner.call("run", seqs, False)
    print(f"  -> run 返回 {toks}")
    llm.scheduler.postprocess(seqs, toks, False)
    print(f"  -> 之后 token_ids 末尾: {s.token_ids[-6:]}")
    step += 1

print(f"\n最终: {seqs[0].completion_token_ids}")
