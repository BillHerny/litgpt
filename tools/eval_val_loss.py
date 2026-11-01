"""Compare validation loss of the base vs LoRA-finetuned Qwen2.5-1.5B on Alpaca2k val split.

Usage:
    python tools/eval_val_loss.py base checkpoints/Qwen/Qwen2.5-1.5B
    python tools/eval_val_loss.py finetuned out/qwen2.5-1.5b-qlora/step-000200
"""

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
