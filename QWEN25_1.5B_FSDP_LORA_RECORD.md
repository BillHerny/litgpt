# Qwen2.5-1.5B 双卡 FSDP + LoRA 微调全记录

> 环境：AutoDL 容器 / 2 × RTX 2080 Ti / litgpt 0.5.13
> 时间：2026-11-01
> 目的：把「从跑不动到跑通并验证」的每一步、每个报错、每个决策都记录下来，作为学习笔记。

---

## 0. TL;DR

| 项目 | 结果 |
|---|---|
| 最终可用方案 | `litgpt finetune <ckpt> --devices 2 --precision 32-true`（FSDP2 + LoRA r=16） |
| 训练量 | 200 / 237 步（约 84% 一轮），2,000 条 Alpaca2k |
| 验证集 loss | 1.3334 → **0.9401**（−29.5%） |
| 验证集 ppl | 3.79 → **2.56**（−32.5%） |
| 显存 | 9.1–9.4 GB / 卡（fp32 分片） |
| 速度 | 2.70 s/iter，约 10.8 s/step（global batch 8） |
| 产物 | `out/qwen2.5-1.5b-qlora/step-000200/lit_model.pth`（合并后 6.7GB）+ `.lora`（8.4MB） |

原命令里有三处**在当前环境下必然失败**的东西：

1. `--quantize bnb.nf4` + `--devices 2`：litgpt 源码直接 `raise NotImplementedError`；
2. `--precision 16-true`：纯 fp16 无梯度缩放，loss 第 2 步就 NaN；
3. `litgpt finetune lora ... --checkpoint_dir ...`：0.5.13 里没有 `lora` 子命令，`checkpoint_dir` 是位置参数。

再加上环境本身的 torch/驱动不匹配，以及两个 NCCL 死锁，一共踩了 8 个坑。

---

## 1. 硬件与软件环境

### 1.1 硬件

```
CPU：24 核
内存：80 GB
GPU：2 × NVIDIA GeForce RTX 2080 Ti（11264 MiB / 卡，sm_75，Turing）
驱动：550.90.07（支持到 CUDA 12.4）
系统盘：/ 共 30 GB（初始仅剩 6.2 GB）
数据盘：/root/autodl-tmp 50 GB（未使用）
```

### 1.2 Python 环境

```
解释器：/root/miniconda3/bin/python 3.12.3
虚拟环境：/root/code/litgpt-main/my_project_env（venv，不继承系统包）
litgpt：0.5.13，安装在 venv 的 site-packages（非 editable）
```

最终关键版本：

| 包 | 版本 |
|---|---|
| torch | **2.8.0+cu126** |
| lightning / pytorch-lightning | 2.6.6 |
| litgpt | 0.5.13 |
| bitsandbytes | 0.46.1 |
| datasets | 3.6.0 |
| tokenizers | 0.23.3 |
| huggingface_hub | 1.3.7 |
| numpy | 2.5.4 |
| triton | 3.4.0 |
| pyarrow / pandas | 26.0.0 / 3.1.0 |

### 1.3 网络

```
huggingface.co       → 连接超时（curl exit 28）
hf-mirror.com        → 可用
pip 默认源            → 阿里云 mirrors.aliyun.com
download.pytorch.org → 可用
```

---

## 2. 模型与数据准备

### 2.1 模型

`checkpoints/Qwen/Qwen2.5-1.5B/`（已提前用 `litgpt convert_hf_checkpoint` 转换过）：

| 文件 | 大小 | 说明 |
|---|---|---|
| `lit_model.pth` | 2.9 GB | litgpt 格式权重（训练真正读取的文件） |
| `model.safetensors` | 2.9 GB | 原始 HF 权重（与上面冗余，后期已删除） |
| `config.json` / `model_config.yaml` | 684 B / 1.1 KB | HF / litgpt 两套配置 |
| `tokenizer.json` / `tokenizer_config.json` | 6.8 MB | 分词器 |
| `generation_config.json` | 138 B | 生成参数 |

模型结构（来自 `model_config.yaml`）：

```
n_layer=28, n_embd=1536, n_head=12, n_query_groups=2 (GQA)
intermediate_size=8960, vocab_size=151936, rope_base=1e6
block_size=131072, norm=RMSNorm(eps=1e-6), mlp=LLaMAMLP
```

### 2.2 数据

- 数据集：`mhenrichsen/alpaca_2k_test`（HF 上的单 parquet 文件，1.7 MB，2000 条）
- 通过 `HF_ENDPOINT=https://hf-mirror.com` 下载，缓存在 `data/alpaca2k`
- 划分：`val_split_fraction=0.05` → 训练 1900 条 / 验证 100 条
- prompt 模板：`alpaca`；`mask_prompt=False`（loss 包含 prompt 部分）
- 训练数据最长序列：**612**（日志：`longest sequence length ... is 612`）

---

## 3. 完整时间线

| 时间 | 事件 |
|---|---|
| 15:34 | 下载 Qwen2.5-1.5B 原始文件 |
| 15:49 | `lit_model.pth` 转换完成 |
| 19:24 | 发现 `torch.cuda.is_available() == False` |
| 19:27 | 卸载 CUDA 13 版 torch，开始安装 `torch 2.8.0+cu126` |
| 19:47 | torch 安装完成，双卡 CUDA 验证通过 |
| 19:48 | 安装 `datasets` |
| 19:49 | HF 镜像加载数据集成功（2000 条） |
| 19:50 | 第 1 次训练（`16-true`）→ iter 8 loss NaN，立即停止 |
| 19:52 | 第 2 次尝试（`16-mixed`）→ Lightning 直接报 `ValueError` |
| 19:53 | 第 3 次训练（`32-true`）启动 |
| ~20:12 | step 100 触发验证 → rank0 生成示例，rank1 在 barrier → 死锁 |
| 20:22:26 | NCCL watchdog 600s 超时，rank1 崩溃 |
| 20:23 | 进程被终止，本次无任何 checkpoint 落盘 |
| 20:24:51 | 第 4 次训练启动（`--eval.interval 10000 --train.save_interval 100`） |
| 20:43:15 | `step-000100` 保存 |
| 21:01:27 | `step-000200` 保存 |
| ~21:08 | iter 950，迭代器绕回 epoch → 再次进入 `generate_example` 分支 → 死锁 |
| ~21:18 | NCCL 超时（推断），进程被杀，**final 保存与合并未执行** |
| 21:26 | 手动合并 `step-000200` → `lit_model.pth`（6.7 GB，fp32） |
| 21:29 | 验证集 loss 对比 + 生成对比 |

---

## 4. 问题排查实录

### 4.1 第一关：torch 与显卡驱动不匹配（最根本）

**现象**

```text
torch 2.15.0+cu132
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12040). Please update your GPU driver ...
cuda False
n_gpu 2
```

**原因**

- `torch 2.15.0+cu132` 是针对 **CUDA 13.2** 编译的；
- CUDA 的「小版本兼容」只在同一大版本内成立（12.x ↔ 12.x），跨大版本需要新驱动；
- CUDA 13 需要 R580 及以上驱动，而机器是 `550.90.07`（最高 CUDA 12.4）；
- 于是哪怕 `device_count()==2`，CUDA 上下文也创建失败，所有 GPU 运算不可用。

> 这也解释了最早 `from litgpt import LLM` 退出码 2：任何 CUDA 初始化都会失败。

**处理**

```bash
# 1) 卸载 CUDA 13 版本的整套包（同时释放系统盘，当时只剩 6.2GB）
PKGS=$(./my_project_env/bin/python -m pip list --format=freeze \
  | grep -Ei '^(torch==|torchvision==|torchaudio==|triton==|nvidia-)' \
  | cut -d= -f1 | tr '\n' ' ')
./my_project_env/bin/python -m pip uninstall -y $PKGS

# 2) 安装与驱动兼容、且 CI 验证过的版本（torch 2.8.0，cu126）
./my_project_env/bin/python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url http://mirrors.aliyun.com/pypi/simple \
  "torch==2.8.0+cu126"
```

- 选 `2.8.0`：仓库 CI（`.lightning/workflows/tests.yaml`）里 `TORCH_VERSION: "2.8.0"`；
- 选 `cu126`：向下兼容 CUDA 12.4 驱动；且 cu126 wheel 包含 sm_75 内核（2080 Ti 可用）；
- `--index-url` 指官方 CUDA 源拿 `+cu126`，`--extra-index-url` 指阿里云补齐 `filelock` 等普通依赖。

**验证**

```text
torch 2.8.0+cu126
cuda_available True
GPU0: NVIDIA GeForce RTX 2080 Ti 10.7 GB sm_75
GPU1: NVIDIA GeForce RTX 2080 Ti 10.7 GB sm_75
matmul ok 16.0
```

### 4.2 第二关：litgpt 禁止「多卡 + 量化」

原命令同时包含 `--devices 2` 和 `--quantize bnb.nf4`。源码里是硬性禁止：

```python
# litgpt/finetune/lora.py（setup 中）
if devices * num_nodes > 1:
    if quantize:
        raise NotImplementedError(
            "Quantization is currently not supported for multi-GPU training. "
            "Please set devices=1 and num_nodes=1 when using the --quantize flag."
        )
```

**为什么**：量化用的是 `lightning.fabric.plugins.BitsandbytesPrecision`，它是**单设备插件**；而 FSDP 需要对参数做 all-gather / reshard，量化后的 4-bit 存储格式与 FSDP 的分片机制不兼容。

**决策**：保留「双卡 FSDP」，去掉量化。
理由：1.5B fp32 权重约 7.1GB，FSDP 全分片后每卡约 3.6GB，11GB 完全够；NF4 在这里没有必要。

### 4.3 第三关：CLI 语法（0.5.13 变了）

**现象**

```text
error: Unrecognized arguments: --checkpoint_dir checkpoints/Qwen/Qwen2.5-1.5B
error: Unrecognized arguments: checkpoints/Qwen/Qwen2.5-1.5B
```

**原因**

`litgpt/__main__.py` 中：

```python
PARSER_DATA = {
    "finetune": finetune_lora_fn,   # finetune 直接就是 LoRA 微调
    ...
}
```

`checkpoint_dir` 是**位置参数**。写 `finetune lora <path>` 时，`lora` 被当成了 `checkpoint_dir`，真正的路径就成了多余参数。

**正确写法**

```bash
litgpt finetune <checkpoint_dir> [其他参数]
```

### 4.4 第四关：数据依赖 + 网络 + 磁盘

1. **`datasets` 缺失**
   `--data Alpaca2k` 的 `prepare_data()` 调用 `datasets.load_dataset`，但 venv 里没装（litgpt 的 `[extra]` 当时只装了一部分）。

   ```bash
   ./my_project_env/bin/python -m pip install --no-cache-dir "datasets>=2.18,<4"
   # 实际安装 datasets 3.6.0 + pyarrow 26.0.0 + pandas 3.1.0
   ```

2. **huggingface.co 连不上**

   ```text
   curl: (28) Failed to connect to huggingface.co port 443 after 6225 ms: Connection timed out
   hf-mirror: 200
   ```

   解决：所有命令前加 `HF_ENDPOINT=https://hf-mirror.com`。
   并额外把 `alpaca_2000.parquet` 预下载到 `data/alpaca2k_local/` 作为兜底。

3. **磁盘紧张**
   系统盘 30GB，任务开始时只剩 6.2GB。做法：
   - 先卸载 6.7GB 的 CUDA 13 包（torch/triton/nvidia-*），腾到 8.9GB；
   - 安装全程 `--no-cache-dir`；
   - 后期删掉与 `lit_model.pth` 重复的 `model.safetensors`（2.9GB）。

   可用 `litgpt convert_from_litgpt` 从 `lit_model.pth` 重新生成 HF 权重，因此该删除可逆。

### 4.5 第五关：精度选择（本次最绕的地方）

| 尝试 | 结果 | 原因 |
|---|---|---|
| `--precision 16-true` | iter 8 起 loss 全 NaN | 纯 fp16：参数/梯度/loss 都是 fp16，没有 GradScaler、没有 fp32 主权重，梯度溢出 |
| `--precision 16-mixed` | 启动即报错 | `ModelParallelStrategy` 不支持 16-mixed |
| `--precision bf16-*` | 不可用 | 2080 Ti 是 Turing（sm_75），**没有 bf16 硬件支持** |
| `--precision 32-true` | ✅ 稳定 | 参数与计算均 fp32，不会溢出 |

失败时的日志（先是正常，然后突然全 NaN）：

```text
iter 7 step 1 | loss train: 1.726
iter 8 step 2 | loss train: nan
iter 9 step 2 | loss train: nan
```

`16-mixed` 的报错原文：

```text
ValueError: The `ModelParallelStrategy` does not support `Fabric(..., precision='16-mixed')`.
Choose a different precision among: 32-true, bf16-mixed, bf16-true, 16-true.
```

**32-true 为什么放得下**（估算）：

```
参数：1.78B × 4B = 7.1GB  → FSDP 分 2 卡 ≈ 3.6GB/卡
梯度：只有 LoRA 参数（2.18M）需要，几乎为 0
优化器：AdamW 只作用于 LoRA 参数，也可忽略
激活：litgpt 的 parallelize_fn 默认开 activation checkpointing，micro-batch=1、seq≤612
实测峰值：9.1–9.4GB/卡
```

**代价**：Turing 的 fp32 没有 TF32/张量核加速，约 2.70 s/iter。

### 4.6 第六关：NCCL 死锁（第一次，验证阶段）

step 100 时 `--eval.interval` 命中，日志停在：

```text
Validating ...
Generate a funny caption for the following photo.
[rank1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2, OpType=ALLREDUCE,
        NumelIn=1, Timeout(ms)=600000) ran for 600049 milliseconds before timing out.
...
[rank: 1] Child process with PID 78359 terminated with code -6.
```

对应源码：

```python
# litgpt/finetune/lora.py
if not is_accumulating and step_count % eval.interval == 0:
    val_loss = validate(fabric, model, val_dataloader, eval)   # 所有 rank 参与
    if fabric.global_rank == 0:
        generate_example(fabric, model, tokenizer, eval, data)  # 仅 rank0
    fabric.barrier()                                            # rank1 在这里等
```

**死锁链条**

1. `generate_example` 用 FSDP 分片后的模型逐 token 前向；
2. 每一层 forward 都需要 **all-gather**，而 all-gather 必须 rank0 + rank1 同时参与；
3. rank1 此时已经走到 `fabric.barrier()`（NCCL 里实现为一次 1 元素 all-reduce）；
4. rank0 等 rank1 参与 all-gather，rank1 等 rank0 参与 barrier → 互等；
5. 600 秒后 watchdog 报 timeout，进程被 `-6`（SIGABRT）终止。

**规避**（不改源码）：把周期验证关掉。

```bash
--eval.interval 10000    # 大于总步数，日志中其它 rank 不会进入该分支
```

最终验证只调用 `validate()`（全 rank 参与的 forward），不会生成示例，因此安全。

### 4.7 第七关：epoch 边界死锁（导致最终保存丢失）

第二次运行跑到 iter 950（约 step 237）时再次崩溃，而且 `final/` 目录不存在。

原因在 `fit()`：

```python
max_steps = train.max_steps or float("inf")   # 只给 --train.epochs 时是 inf

while step_count < max_steps:                  # 永远成立
    batch = next(train_iterator)               # CycleIterator，epoch 属性在绕回时 +1
    if train_iterator.epoch >= train.epochs:
        if fabric.global_rank == 0:
            generate_example(...)              # ← 又是它
        break
```

也就是说：**只给 `--train.epochs` 时，退出循环的唯一路径就是 epoch 边界分支，而这个分支必然调用 `generate_example`**，在 FSDP 下必然死锁。

后果：

- 训练本身已经跑完（iter 950 ≈ 237 步）；
- 但 `main()` 末尾的 `save_lora_checkpoint(final)` 和 `merge_lora` 都没执行；
- 只能退回到 `step-000200` 的适配器（84% 训练量）。

**修复**：显式给有限步数，让 `while` 条件在迭代器绕回前失效：

```bash
--train.max_steps 236
```

另外建议 `--train.save_interval 100`，保证中途有可回退的产物。

> 彻底修法（可选）：把 `generate_example` 的调用改为
> `if fabric.global_rank == 0 and fabric.world_size == 1:`，
> 或让所有 rank 一起执行生成。

### 4.8 第八关：合并模型与磁盘

```bash
litgpt merge_lora out/qwen2.5-1.5b-qlora/step-000200 \
  --pretrained_checkpoint_dir checkpoints/Qwen/Qwen2.5-1.5B \
  --precision 16-true
```

注意：**即使指定 `16-true`，merge_lora 仍输出 fp32 的 `lit_model.pth`（6.7GB）**。
当时磁盘只剩 599MB，处理方式是删除 `checkpoints/Qwen/Qwen2.5-1.5B/model.safetensors`（与 `lit_model.pth` 权重重复），恢复到 3.5GB 可用。

推理时 litgpt 会按 `--precision` 再把权重转成 fp16，所以磁盘上的 fp32 不影响显存。

---

## 5. 最终训练配置（逐参数解释）

```bash
export HF_ENDPOINT=https://hf-mirror.com

./my_project_env/bin/litgpt finetune checkpoints/Qwen/Qwen2.5-1.5B \
  --out_dir out/qwen2.5-1.5b-qlora \
  --devices 2 \
  --precision 32-true \
  --data Alpaca2k \
  --lora_r 16 \
  --lora_alpha 32 \
  --train.micro_batch_size 1 \
  --train.global_batch_size 8 \
  --train.epochs 1 \
  --train.save_interval 100 \
  --eval.interval 10000
```

| 参数 | 作用 | 说明 |
|---|---|---|
| `finetune <dir>` | 选择 LoRA 微调 | 0.5.13 无 `lora` 子命令 |
| `--devices 2` | 双卡 | 内部走 `ModelParallelStrategy`（FSDP2，data_parallel_size=2） |
| `--precision 32-true` | fp32 | 16-true 会 NaN；16-mixed 不被支持；无 bf16 |
| `--data Alpaca2k` | 内置数据模块 | 2000 条，5% 验证 |
| `--lora_r 16 / --lora_alpha 32` | LoRA 秩与缩放 | 实际缩放 = alpha/r = 2 |
| `--train.micro_batch_size 1` | 每卡每 iter 1 条 | 显存安全 |
| `--train.global_batch_size 8` | 全局 batch | 梯度累积 = 8 / (2 卡 × 1) = 4 |
| `--train.epochs 1` | 1 轮 | 本目录下用 epochs 会踩 4.7 的死锁 |
| `--train.save_interval 100` | 每 100 步存一次 | 崩溃后可回退 |
| `--eval.interval 10000` | 关闭周期验证 | 避开 4.6 的死锁；最终验证仍执行 |

其他默认值（来自日志）：

```
lora_dropout=0.05, lora_query=True, lora_value=True（key=False, mlp/proj/head=False）
optimizer=AdamW, lr_warmup_steps=100（峰值约 1e-3，之后余弦衰减）, min_lr=6e-5
eval.max_iters=100, eval.max_new_tokens=100, final_validation=True
logger=csv, seed=1337, num_workers=4/rank
```

LoRA 参数量（2,179,072）的构成：

```
q_proj 权重 1536×1536，v_proj 权重 256×1536（GQA: 2 个 KV head × 128）
每层 LoRA：q: 16×1536 + 1536×16 = 49,152
          v: 16×1536 +  256×16 = 28,672
          合计 77,824 / 层
28 层 × 77,824 = 2,179,072
```

训练日志确认：

```text
Number of trainable parameters: 2,179,072
Number of non-trainable parameters: 1,777,088,000
```

---

## 6. 训练过程数据

### 6.1 规模换算

```
训练样本 1900（95%）       每卡 dataloader 950 条
global_batch_size 8        梯度累积 4 iters
max_steps ≈ 950 // 4 = 237
总 iter ≈ 950
```

### 6.2 loss 抽样（单 iter loss 有噪声，看趋势）

| iter | loss | lr |
|---|---|---|
| 1 | 1.268 | 0（warmup 起点） |
| 15 | 0.911 | ~1.5e-4 |
| 50 | 0.978 | ~5e-4 |
| 200 | 0.891 | ~1e-3 |
| 500 | ~0.9–1.3 波动 | 衰减中 |
| 950 | 1.287 | ~0 |

### 6.3 吞吐与资源

| 指标 | 数值 |
|---|---|
| iter_time | 2.69–2.75 s（偶发 3.0s） |
| 每 step 时间 | ≈ 10.8 s（4 iters） |
| 样本吞吐 | ≈ 0.74 样本/s |
| token 吞吐 | ≈ 131 token/s（累计 336,476 tokens / 2565 s） |
| 显存 | 9.1–9.4 GB / 11.0 GB |
| GPU 利用率 | 98–99% |
| 实测总时长 | ≈ 43 分钟（950 iters） |

---

## 7. 效果验证

### 7.1 定量：验证集 loss（100 条，训练全程未参与更新）

评估脚本见附录 A，同一 tokenizer / prompt 模板 / max_seq_length=612 / `mask_prompt=False`。

| 模型 | val_loss | val_ppl |
|---|---|---|
| Qwen2.5-1.5B（原始） | 1.3334 | 3.79 |
| 微调后（step-200） | **0.9401** | **2.56** |
| 变化 | −0.3933（**−29.5%**） | −32.5% |

### 7.2 定性：同一指令对比

指令：`### Instruction:\nName three benefits of regular exercise.\n\n### Response:\n`

**原始模型**（base 模型，不识别 Alpaca 格式）：

```text
撙报sed
撙报user
撙
```

**微调后**：

```text
### Response:

### Response:
1. Improved Heart Health: Regular exercise helps lower blood pressure,
   improve cholesterol levels, and reduce the risk of heart disease...
```

### 7.3 结论与边界

1. 训练有效：同分布验证集 loss −29.5%，格式遵循与连贯性明显改善。
2. 收益主要来自**风格/格式适配**：2k 样本 + 200 步不足以改变模型知识能力。
3. 验证集只有 100 条且与训练集同源，数值只能说明「拟合了该数据分布」，不代表通用能力。
4. 未完成完整一轮（200/237 步）。

---

## 8. 产物与使用

```
out/qwen2.5-1.5b-qlora/
├── logs/csv/version_0/metrics.csv      # 训练指标
├── step-000100/                        # 第 100 步
│   ├── lit_model.pth.lora (8.7 MB)
│   └── config/tokenizer/hyperparameters...
└── step-000200/                        # 第 200 步（本次使用）
    ├── lit_model.pth        (6.7 GB)   # 已合并 LoRA 的完整模型（fp32）
    ├── lit_model.pth.lora   (8.4 MB)   # LoRA 适配器
    ├── hyperparameters.yaml
    └── config/tokenizer/prompt_style...
```

### 8.1 用合并后的模型生成

```bash
litgpt generate out/qwen2.5-1.5b-qlora/step-000200 \
  --precision 16-true --max_new_tokens 200 \
  --prompt $'### Instruction:\n用三句话解释什么是 LoRA。\n\n### Response:\n'
```

### 8.2 用适配器（不占额外磁盘）

```bash
litgpt generate checkpoints/Qwen/Qwen2.5-1.5B \
  --adapter out/qwen2.5-1.5b-qlora/step-000200 \
  --precision 16-true --max_new_tokens 200 \
  --prompt $'### Instruction:\n用三句话解释什么是 LoRA。\n\n### Response:\n'
```

### 8.3 单独合并

```bash
litgpt merge_lora <adapter_dir> \
  --pretrained_checkpoint_dir checkpoints/Qwen/Qwen2.5-1.5B \
  --precision 16-true
# 注意：输出仍是 fp32 的 <adapter_dir>/lit_model.pth（约 6.7GB）
```

---

## 9. 原理笔记

### 9.1 为什么 cu132 的 torch 在 CUDA 12.4 驱动上不能用

- CUDA「minor version compatibility」：CUDA 12.x 程序可运行在任意支持 12.x 的驱动上（R525+）；
- 但大版本不兼容：CUDA 13 程序需要 R580+；
- PyTorch wheel 自带 CUDA runtime，因此驱动必须 ≥ wheel 的 CUDA 大版本要求。

### 9.2 FSDP2（`fully_shard`）做了什么

- 把每个参数按 data-parallel rank **切片存储**（shard），前向时按层 **all-gather** 成完整参数，用完即释放；
- 反向时再 all-gather，计算完对梯度做 **reduce-scatter**；
- 好处：参数、梯度、优化器状态显存都除以卡数；
- 代价：层内计算需要集合通信，任何 rank 不参与都会卡死（见 4.6 / 4.7）。

litgpt 的实现要点：

```python
# litgpt/finetune/lora.py
for m in reversed(list(model.modules())):
    if (isinstance(m, torch.nn.Linear) and m.weight.requires_grad) \
       or isinstance(m, CheckpointWrapper) or isinstance(m, Block):
        fully_shard(m, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

注意条件：只对 `weight.requires_grad` 的 Linear 做分片 —— 因为 LoRA 冻结了原权重，所以这里实际上是「按 Block 分片 + 对可训练 Linear 分片」的组合。

### 9.3 LoRA 数学形式

原权重 `W` 冻结，学习低秩增量：

$$
W' = W + \frac{\alpha}{r} \cdot B A, \qquad A \in \mathbb{R}^{r \times d_{in}},\; B \in \mathbb{R}^{d_{out} \times r}
$$

本实验 `r=16, α=32` → 缩放系数 `α/r = 2`。可训练参数只占全模型的 0.12%。

### 9.4 fp16 为什么会 NaN

- fp16 最大约 65504，最小正规数约 6e-5；
- 反向传播的梯度常常小于 6e-5 → 下溢为 0（学不动），或累加后上溢为 inf → NaN；
- 混合精度（`16-mixed`）通过 **GradScaler**（放大 loss 再缩回梯度）和 **fp32 主权重**解决这个问题；
- `16-true` 两者都没有，因此对 28 层 Transformer 训练极易溢出；
- bf16 指数范围与 fp32 相同，不需要 GradScaler，但 Turing 卡没有 bf16 支持。

### 9.5 NCCL 集合通信与死锁

- `all-gather` / `all-reduce` / `barrier` 都是**集合通信**：同一通信组内所有 rank 必须按相同顺序、相同参数调用；
- 只让 rank0 走一次「需要全部 rank 参与」的计算（如 FSDP generation），其余 rank 走到别的集合通信（如 barrier），就会互相等待；
- NCCL watchdog 默认 600s（10 分钟）超时后强制终止进程。

---

## 10. 踩坑清单（速查版）

1. **驱动决定 CUDA 大版本**：`torch==...+cuXXX` 的 CUDA 版本不能超过驱动的最大支持版本。
2. **`torch.cuda.is_available()` False 时先看驱动**，不要怀疑显卡。
3. **litgpt 0.5.13：`finetune` 就是 LoRA**，`checkpoint_dir` 是位置参数。
4. **多卡 + `--quantize` 不支持**（FSDP 与 bnb 4bit 存储不兼容）。
5. **`ModelParallelStrategy` 只接受 32-true / bf16-mixed / bf16-true / 16-true**。
6. **`16-true` 训练会 NaN**，能用 bf16 就用 bf16，否则用 32-true。
7. **多卡 FSDP 下不要只让 rank0 生成文本**（会死锁）。
8. **只给 `--train.epochs` 时，epoch 边界也会触发上述生成**，请显式 `--train.max_steps`。
9. **`--eval.interval` 要大于总步数**，否则周期验证就死锁。
10. **`--train.save_interval` 一定要设**，崩溃后还能回退。
11. **HF 被墙用 `HF_ENDPOINT=https://hf-mirror.com`**。
12. **`merge_lora` 输出是 fp32**，磁盘要留约 7GB。
13. **系统盘只有 30GB**，安装大 wheel 前先卸载旧 CUDA 包、全程 `--no-cache-dir`。
14. **日志里成百上千行 `odict_keys([...])` 是 FSDP 加载 checkpoint 时的键名打印，不是错误**。
15. **`All GPUs are fully connected via NVLink.` 是连通性提示，无害**。

---

## 11. 常用命令速查

```bash
# 进入环境
cd /root/code/litgpt-main

# 看训练进度
tail -f out/qwen2.5-1.5b-qlora/logs/csv/version_0/metrics.csv

# 看显存/利用率
nvidia-smi -l 1
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv -l 1

# 看是谁占的显存
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 停止训练
pkill -f "my_project_env/bin/litgpt finetune"

# 验证 CUDA
./my_project_env/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# 完整一轮（避开 epoch 边界死锁）
HF_ENDPOINT=https://hf-mirror.com ./my_project_env/bin/litgpt finetune checkpoints/Qwen/Qwen2.5-1.5B \
  --out_dir out/qwen2.5-1.5b-lora-full \
  --devices 2 --precision 32-true --data Alpaca2k \
  --lora_r 16 --lora_alpha 32 \
  --train.micro_batch_size 1 --train.global_batch_size 8 \
  --train.max_steps 236 --train.save_interval 100 \
  --eval.interval 10000
```

---

## 12. 后续优化方向

| 方向 | 做法 | 预期 |
|---|---|---|
| 完成一轮 | `--train.max_steps 236` 或更多 epoch | 收敛更充分 |
| 增大 LoRA 覆盖面 | `--lora_mlp true --lora_projection true` | 参数量↑，拟合能力↑ |
| 换卡 | 3090/4090/A10/A100 | 可用 bf16，速度提升数倍 |
| 更大数据 | Alpaca-GPT4 / LIMA / 自建指令集 | 泛化更好 |
| 更长上下文 | `--train.max_seq_length 1024` | 当前数据最长 612，暂非瓶颈 |
| 加速 | micro-batch 2（显存有余量时） | 减少累积开销，提升吞吐 |
| 训练稳定性 | `--train.max_norm 1.0` | 梯度裁剪，更稳（需验证兼容性） |
| 评估 | 用 lm-eval-harness 跑 MMLU/ARC | 衡量通用能力是否退化 |

---

## 附录 A：验证集 loss 对比脚本

脚本已保存在仓库内：`tools/eval_val_loss.py`。

```python
"""Compare validation loss of the base vs LoRA-finetuned Qwen2.5-1.5B on Alpaca2k val split."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import lightning as L
import torch

from litgpt.args import EvalArgs
from litgpt.config import Config
from litgpt.data.alpaca_2k import Alpaca2k
from litgpt.finetune.lora import validate
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from litgpt.utils import load_checkpoint

MAX_SEQ = 612


def evaluate(label: str, ckpt_dir: str) -> None:
    ckpt = Path(ckpt_dir)
    fabric = L.Fabric(devices=1, precision="16-true")
    fabric.launch()

    config = Config.from_file(ckpt / "model_config.yaml")
    tokenizer = Tokenizer(ckpt)
    with fabric.init_module(empty_init=False):
        model = GPT(config)
    load_checkpoint(fabric, model, ckpt / "lit_model.pth", strict=False)
    model = fabric.setup_module(model)
    model.max_seq_length = MAX_SEQ

    data = Alpaca2k(download_dir=Path("data/alpaca2k"), num_workers=2)
    data.connect(tokenizer=tokenizer, batch_size=1, max_seq_length=MAX_SEQ)
    data.prepare_data()
    data.setup()

    # 关键：必须用 fabric.setup_dataloaders 把 batch 搬到 GPU
    val_dataloader = fabric.setup_dataloaders(data.val_dataloader())
    eval_args = EvalArgs(interval=1, max_new_tokens=100, max_iters=100)
    loss = validate(fabric, model, val_dataloader, eval_args, verbose=False)
    print(
        f"RESULT|{label}|val_loss={float(loss):.4f}"
        f"|val_ppl={torch.exp(loss).item():.2f}"
        f"|val_samples={len(val_dataloader.dataset)}"
    )


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
```

运行：

```bash
python tools/eval_val_loss.py base checkpoints/Qwen/Qwen2.5-1.5B
python tools/eval_val_loss.py finetuned out/qwen2.5-1.5b-qlora/step-000200
```

> 小坑：如果不用 `fabric.setup_dataloaders`，batch 会留在 CPU，
> 报 `Expected all tensors to be on the same device ... index_select`。

---

## 附录 B：关键报错原文

```text
# 1) 驱动过旧
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12040).

# 2) 多卡 + 量化
NotImplementedError: Quantization is currently not supported for multi-GPU training.
Please set devices=1 and num_nodes=1 when using the --quantize flag.

# 3) CLI
error: Unrecognized arguments: --checkpoint_dir checkpoints/Qwen/Qwen2.5-1.5B

# 4) 精度不支持
ValueError: The `ModelParallelStrategy` does not support `Fabric(..., precision='16-mixed')`.
Choose a different precision among: 32-true, bf16-mixed, bf16-true, 16-true.

# 5) NCCL 死锁
[Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2, OpType=ALLREDUCE,
NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600049 milliseconds before timing out.
[rank: 1] Child process with PID 78359 terminated with code -6.
```

---

## 附录 C：本次未采用的方案

| 方案 | 为什么没采用 |
|---|---|
| 单卡 + `bnb.nf4` QLoRA | 与你的「双卡 FSDP」目标冲突；且 1.5B 在 11GB 上无需量化 |
| `bf16-true` | 2080 Ti（sm_75）无 bf16 硬件支持 |
| `16-mixed` | FSDP2 策略不支持 |
| 2 卡 DDP（非 FSDP） | litgpt 0.5.13 多卡仅提供 `ModelParallelStrategy` |
| 修改源码跳过 `generate_example` | 可以通过参数规避，尽量不动第三方包 |
| 使用 `model.safetensors` 作为训练权重 | litgpt 训练读取 `lit_model.pth` |

---

*本文档记录于 2026-11-01，对应 litgpt 0.5.13 / torch 2.8.0+cu126 / 2×RTX 2080 Ti。*
