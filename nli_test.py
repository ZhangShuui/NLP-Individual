"""multi_nli_batch_eval.py
=================================
批量评估 MultiNLI 的两类模型：
1. 🤗 Transformers 本地 / 检查点模型（含 GPU 并行支持）
2. OpenAI ChatCompletion API 模型（如 **gpt-4o**）

用法示例
---------
# 本地 RoBERTa (FP32 推理)
python multi_nli_batch_eval.py \
    --model-name roberta-large-mnli \
    --split dev_matched \
    --batch-size 32

# OpenAI GPT‑4o API（一行命令，需设置 OPENAI_API_KEY）
python multi_nli_batch_eval.py \
    --model-name gpt-4o \
    --split dev_matched \
    --batch-size 20 
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import openai  # type: ignore
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# 全局标签映射：与 MultiNLI 官方保持一致
# ---------------------------------------------------------------------------
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# 默认 System Prompt，指导 GPT-4o 回答三个标签之一
SYSTEM_PROMPT = (
    "You are an NLI classifier. Given a premise and a hypothesis, "
    "respond with one of exactly three words: entailment, neutral, or contradiction."
)

# ---------------------------------------------------------------------------
# Argument Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser("Batch evaluate Local/API models on MultiNLI")
    parser.add_argument("--model-name", type=str, required=True, help="Model name or 'gpt-4o'")
    parser.add_argument("--split", type=str, default="dev_matched", choices=[
        "train", "dev_matched", "dev_mismatched"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=16, help="Per‑device batch size")
    parser.add_argument("--sample-size", type=int, default=None, help="Subset for quick tests")
    parser.add_argument("--num-proc", type=int, default=4, help="HF map() workers")
    parser.add_argument("--openai-max-conn", type=int, default=5, help="并发请求数")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Dataset Loading & Pre‑processing
# ---------------------------------------------------------------------------

def load_multinli(split: str, sample_size: int | None, num_proc: int) -> "datasets.Dataset":
    """Load MultiNLI from HF hub and optionally subsample."""
    ds = load_dataset("multi_nli", split=split)
    if sample_size:
        ds = ds.shuffle(seed=42).select(range(sample_size))

    # 统一列名：premise / hypothesis / label
    rename_map = {}
    if "sentence1" in ds.column_names:
        rename_map["sentence1"] = "premise"
        rename_map["sentence2"] = "hypothesis"
    if rename_map:
        ds = ds.rename_columns(rename_map)
    return ds

# ---------------------------------------------------------------------------
# Local HuggingFace Model Inference
# ---------------------------------------------------------------------------

def tokenize_function(examples, tokenizer):
    """Tokenize with pair input."""
    return tokenizer(
        examples["premise"], examples["hypothesis"], truncation=True, max_length=256
    )


def eval_local_model(ds, model_name: str, batch_size: int, num_proc: int):
    """Evaluate HF model checkpoint locally (CUDA / DataParallel 支持)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize dataset (num_proc workers -> fast)
    ds_tok = ds.map(tokenize_function, batched=True, num_proc=num_proc, fn_kwargs={"tokenizer": tokenizer})
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Collation: dynamic padding to max length of batch
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    dataloader = DataLoader(ds_tok, batch_size=batch_size, collate_fn=collator)

    # Load model (sequence classification head) and push to device(s)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    preds, refs = [], []
    for batch in tqdm(dataloader, desc="Inference"):
        batch = {k: v.to(device) for k, v in batch.items() if k != "label"}
        with torch.no_grad():
            logits = model(**batch).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        refs.extend(batch["labels"].cpu().tolist() if "labels" in batch else [])

    return compute_metrics(preds, refs)

# ---------------------------------------------------------------------------
# OpenAI GPT‑4o API Inference
# ---------------------------------------------------------------------------
async def _call_chat_api(messages: List[dict], model: str, retry: int = 3):
    """Single request with exponential back‑off重试."""
    for attempt in range(retry):
        try:
            resp = await openai.ChatCompletion.acreate(model=model, messages=messages)
            return resp["choices"][0]["message"]["content"].strip().lower()
        except Exception as e:
            if attempt == retry - 1:
                raise e
            await asyncio.sleep(2 ** attempt)


def chunk(iterable: Sequence, size: int) -> Iterable[Sequence]:
    """Yield successive size‑d chunks."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


async def predict_openai_async(premises: List[str], hypotheses: List[str], model: str, max_conn: int):
    """并发请求 OpenAI ChatCompletion，返回字符串标签。"""
    semaphore = asyncio.Semaphore(max_conn)

    async def _worker(premise: str, hypothesis: str):
        async with semaphore:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer with one word."
                    ),
                },
            ]
            return await _call_chat_api(messages, model=model)

    tasks = [_worker(p, h) for p, h in zip(premises, hypotheses)]
    return await asyncio.gather(*tasks)


def eval_openai(ds, model_name: str, batch_size: int, max_conn: int):
    """Batch evaluate OpenAI model via ChatCompletion API."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")

    preds, refs = [], []
    for batch in tqdm(list(chunk(range(len(ds)), batch_size)), desc="API batches"):
        prem_batch = [ds[i]["premise"] for i in batch]
        hyp_batch = [ds[i]["hypothesis"] for i in batch]
        labels_batch = [ds[i]["label"] for i in batch]

        loop = asyncio.get_event_loop()
        raw_outputs = loop.run_until_complete(
            predict_openai_async(prem_batch, hyp_batch, model=model_name, max_conn=max_conn)
        )

        # Map string answers back to id (default to neutral = 1 if unknown)
        preds.extend([LABEL2ID.get(o.strip(), 1) for o in raw_outputs])
        refs.extend(labels_batch)

    return compute_metrics(preds, refs)

# ---------------------------------------------------------------------------
# Utility: Metric Computation & Pretty Print
# ---------------------------------------------------------------------------

def compute_metrics(preds: List[int], refs: List[int]):
    """Accuracy & macro‑F1."""
    acc = accuracy_score(refs, preds)
    f1 = f1_score(refs, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def save_preds(preds: List[int], ds, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("premise\thypothesis\tpred_label\n")
        for p, item in zip(preds, ds):
            f.write(
                f"{item['premise'].replace('\t', ' ')}\t{item['hypothesis'].replace('\t', ' ')}\t{ID2LABEL[p]}\n"
            )

# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ds = load_multinli(args.split, args.sample_size, args.num_proc)

    if args.model_name.lower().startswith("gpt-"):
        res = eval_openai(ds, args.model_name, args.batch_size, args.openai_max_conn)
    else:
        res = eval_local_model(ds, args.model_name, args.batch_size, args.num_proc)

    # 拓展：可保存预测，或写入 CSV 供后续分析
    print("=== Evaluation Results ===")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
