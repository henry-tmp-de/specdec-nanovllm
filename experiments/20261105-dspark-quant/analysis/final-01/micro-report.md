# DSpark W8A16 layer0 算子 micro

真实 checkpoint 权重、合成 BF16 激活；仅算子时间，不代表整模型或请求加速。
时间为 CUDA Event 每调用均值；BF16/INT8 每轮配对后取中位数及 min–max，范围不是置信区间。

测量行 30，失败 0，解析错误 0。

| 权重 | B | M | 模式 | BF16 ms | INT8 ms | BF16/INT8 median [min,max] | 量化参照 RMS | 数值 |
|---|---:|---:|---|---:|---:|---|---:|---|
| layers.0.mlp.gate_proj.weight | 1 | 7 | eager | 0.07381 | 0.07958 | 0.9276 [0.9238, 0.9476] | 0.001661 | True |
| layers.0.mlp.gate_proj.weight | 1 | 7 | graph | 0.07258 | 0.05053 | 1.438 [1.394, 1.439] | 0.001661 | True |
| layers.0.mlp.gate_proj.weight | 8 | 56 | eager | 0.07212 | 0.1326 | 0.5433 [0.5432, 0.5443] | 0.001657 | True |
| layers.0.mlp.gate_proj.weight | 8 | 56 | graph | 0.07053 | 0.1296 | 0.5443 [0.5323, 0.5448] | 0.001657 | True |
| layers.0.mlp.gate_proj.weight | 16 | 112 | eager | 0.1273 | 0.2028 | 0.6284 [0.6248, 0.629] | 0.00166 | True |
| layers.0.mlp.gate_proj.weight | 16 | 112 | graph | 0.126 | 0.202 | 0.6236 [0.6219, 0.6236] | 0.00166 | True |
| layers.0.mlp.down_proj.weight | 1 | 7 | eager | 0.08894 | 0.09612 | 0.9226 [0.8054, 0.968] | 0.001679 | True |
| layers.0.mlp.down_proj.weight | 1 | 7 | graph | 0.08747 | 0.04901 | 1.787 [1.736, 1.788] | 0.001679 | True |
| layers.0.mlp.down_proj.weight | 8 | 56 | eager | 0.1201 | 0.1008 | 1.191 [1.148, 1.214] | 0.001662 | True |
| layers.0.mlp.down_proj.weight | 8 | 56 | graph | 0.1191 | 0.09887 | 1.206 [1.204, 1.206] | 0.001662 | True |
| layers.0.mlp.down_proj.weight | 16 | 112 | eager | 0.1314 | 0.1665 | 0.7887 [0.7887, 0.8025] | 0.001661 | True |
| layers.0.mlp.down_proj.weight | 16 | 112 | graph | 0.1317 | 0.165 | 0.7987 [0.7893, 0.8006] | 0.001661 | True |
| layers.0.self_attn.q_proj.weight | 1 | 7 | eager | 0.03704 | 0.07771 | 0.4761 [0.4758, 0.4834] | 0.001659 | True |
| layers.0.self_attn.q_proj.weight | 1 | 7 | graph | 0.03612 | 0.02068 | 1.746 [1.744, 1.748] | 0.001659 | True |
| layers.0.self_attn.q_proj.weight | 8 | 56 | eager | 0.06349 | 0.07895 | 0.8135 [0.7983, 0.8197] | 0.001664 | True |
| layers.0.self_attn.q_proj.weight | 8 | 56 | graph | 0.06181 | 0.04564 | 1.354 [1.354, 1.355] | 0.001664 | True |
| layers.0.self_attn.q_proj.weight | 16 | 112 | eager | 0.06625 | 0.08257 | 0.8028 [0.7861, 0.8034] | 0.001661 | True |
| layers.0.self_attn.q_proj.weight | 16 | 112 | graph | 0.06523 | 0.08201 | 0.7954 [0.7909, 0.7975] | 0.001661 | True |
| layers.0.self_attn.k_proj.weight | 1 | 7 | eager | 0.02139 | 0.07664 | 0.2737 [0.2735, 0.2861] | 0.00166 | True |
| layers.0.self_attn.k_proj.weight | 1 | 7 | graph | 0.01253 | 0.0094 | 1.332 [1.33, 1.335] | 0.00166 | True |
| layers.0.self_attn.k_proj.weight | 8 | 56 | eager | 0.02098 | 0.07974 | 0.2677 [0.2623, 0.2679] | 0.001663 | True |
| layers.0.self_attn.k_proj.weight | 8 | 56 | graph | 0.01492 | 0.01739 | 0.8581 [0.8581, 0.9558] | 0.001663 | True |
| layers.0.self_attn.k_proj.weight | 16 | 112 | eager | 0.0271 | 0.05161 | 0.5224 [0.5189, 0.5266] | 0.001662 | True |
| layers.0.self_attn.k_proj.weight | 16 | 112 | graph | 0.02577 | 0.02662 | 0.9681 [0.9677, 0.9684] | 0.001662 | True |
| layers.0.self_attn.o_proj.weight | 1 | 7 | eager | 0.0541 | 0.0771 | 0.7011 [0.6691, 0.7105] | 0.001663 | True |
| layers.0.self_attn.o_proj.weight | 1 | 7 | graph | 0.053 | 0.02085 | 2.542 [2.54, 2.543] | 0.001663 | True |
| layers.0.self_attn.o_proj.weight | 8 | 56 | eager | 0.05461 | 0.07755 | 0.7048 [0.6862, 0.7218] | 0.001657 | True |
| layers.0.self_attn.o_proj.weight | 8 | 56 | graph | 0.05341 | 0.04157 | 1.285 [1.284, 1.335] | 0.001657 | True |
| layers.0.self_attn.o_proj.weight | 16 | 112 | eager | 0.06544 | 0.07174 | 0.9119 [0.9116, 0.9127] | 0.001658 | True |
| layers.0.self_attn.o_proj.weight | 16 | 112 | graph | 0.06385 | 0.07037 | 0.9073 [0.9063, 0.9073] | 0.001658 | True |

原 BF16 权重误差与 graph/eager 对照仅见 micro-summary.json；失败不填零。
