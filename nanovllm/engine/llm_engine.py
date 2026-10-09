import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, token_hook=None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 把提议器交给 scheduler：token_ids 落地后要立刻入索引，
        # 否则下一步提不出以新 token 结尾的候选
        if getattr(self.model_runner, "spec_proposer", None) is not None:
            self.scheduler.set_spec_proposer(self.model_runner.spec_proposer)
        # token 交付 hook（TTFT/TPOT/ITL 的接入点，见 engine/token_hook.py）。
        # ★ 默认 None = 不挂，热路径只有一句 `is not None` 判断，零开销；
        #   要测延迟时显式传进来：LLM(model, token_hook=TokenDeliveryHook())
        self.token_hook = token_hook
        if token_hook is not None:
            self.scheduler.set_token_hook(token_hook)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if self.token_hook is not None:
            # 入队时刻 = TTFT 的参考点（用户视角：排队也算等）
            self.token_hook.on_request_added(seq)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # prefill 的吞吐口径 = 本步写进 KV 的 prompt token 数（分块 prefill 的中间
        # chunk 也照样算，它们不产出 token 但确实占了算力）。必须在 postprocess
        # 把 num_scheduled_tokens 清零之前取出来。
        num_prompt_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else 0
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # ★ 正式吞吐口径：本步【实际交付的 token IDs 数】，由 postprocess 数出来。
        #   旧写法 `-sum(1 + getattr(seq, "last_accepted", 1))` 不是落地数：
        #   last_accepted 是「上一轮落地了几个」的遗留字段（普通 decode 路径恒为 0），
        #   缺省还会补 1，退化时变成每序列 2。投机一轮交付 k+1 个就记 k+1 个。
        num_delivered = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 符号仍然表示方向（正 = prefill，负 = decode），generate() 的 tqdm 语义不变
        num_tokens = num_prompt_tokens if is_prefill else -num_delivered
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
