#!/usr/bin/env python
"""multi_nli_batch_eval.py —— v2  (2025‑05‑13)
Batch evaluation toolkit for the CSIT6000R Section 2 project.
Supports two tasks:
1. **NLI** on MultiNLI (original Section 2.1)
2. **Hallucination detection** on WikiBio‑GPT3‑Hallucination (Section 2.2)

It can handle **local Hugging Face checkpoints**, **OpenAI ChatCompletion
models (GPT‑4o)**, and **Qwen3‑style causal LMs**.  See CLI examples at the
end of this doc‑string.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Optional OpenAI client (only imported when environment variables exist)
# ---------------------------------------------------------------------------


def _build_openai_clients():
    try:
        import openai  # type: ignore

        if os.getenv("OPENAI_API_KEY"):
            sync_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            async_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return sync_client, async_client
    except ModuleNotFoundError:
        pass
    return None, None


openai_client, openai_async_client = _build_openai_clients()

# ---------------------------------------------------------------------------
# Global label mapping (MultiNLI convention) ────────────────────────────────
# For hallucination detection we map: 0 (truth) → entailment, 1 (halluc) → contradiction
# ---------------------------------------------------------------------------

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Prompt templates for ChatCompletion‑style models
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_DEFAULT = (
    "You are an NLI classifier. Given a premise and a hypothesis, "
    "respond with one of exactly three words: entailment, neutral, or contradiction."
)

PROMPT_TEMPLATES = {
    "1_direct_instruction": (
        "You are an expert natural‑language‑inference assistant.\n"
        "Task: Given a PREMISE and a HYPOTHESIS, decide whether the hypothesis is (A) ENTAILED, (B) NEUTRAL, or (C) CONTRADICTED by the premise.\n"
        'Return exactly one word: "entailment", "neutral", or "contradiction".\n\n'
        "Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
    ),
    "2_chain_of_thought": (
        'You are a logical reasoner.\nFirst think step by step, then output "Answer: <label>" with one of [entailment|neutral|contradiction].\n\n'
        "Premise: {premise}\nHypothesis: {hypothesis}\n\nReasoning:"
    ),
    "6_genre_aware_few_shot": (
        "Task: Natural Language Inference. Decide whether H is ENTAILED, NEUTRAL or CONTRADICTS P.\n"
        "Example 1\nGenre: LETTERS\nP: The garden scheme teaches children the value of the land.\nH: All children love gardening.\nLabel: contradiction\n\n"
        "Now solve:\nGenre: {genre}\nP: {premise}\nH: {hypothesis}\nLabel:"
    ),
}

LABEL_REGEX = re.compile(r"\b(entailment|neutral|contradiction)\b", re.I)

# ---------------------------------------------------------------------------
# Helper ‑ parse ChatCompletion output to canonical label
# ---------------------------------------------------------------------------


def parse_openai_response(resp_text: str, template_key: str) -> str:
    text = resp_text.strip().lower()
    # 1️⃣ Try regex first (robust)
    m = LABEL_REGEX.search(text)
    if m:
        return m.group(1)
    # 2️⃣ Template‑specific fall‑back
    if "answer:" in text:
        return text.split("answer:")[-1].strip().split()[0]
    if "label:" in text:
        return text.split("label:")[-1].strip().split()[0]
    # 3️⃣ First word fallback
    return text.split()[0] if text else ""


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch evaluate NLI or Hallucination datasets")
    p.add_argument("--model-name", required=True, help="Checkpoint or 'gpt-4o'")
    p.add_argument(
        "--task",
        choices=["nli", "halluc"],
        default="nli",
        help="Task type: 'nli' for MultiNLI, 'halluc' for WikiBio hallucination",
    )
    p.add_argument(
        "--split",
        default="dev_matched",
        help="Dataset split or custom .jsonl path",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--openai-max-conn", type=int, default=5)
    p.add_argument(
        "--prompt-template",
        default="1_direct_instruction",
        choices=["system_default"] + list(PROMPT_TEMPLATES.keys()),
    )
    p.add_argument("--save-preds", type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _subsample(ds: Dataset, k: int | None) -> Dataset:
    if k and k < len(ds):
        return ds.shuffle(seed=42).select(range(k))
    return ds


def load_multinli(split: str, sample: int | None, num_proc: int) -> Dataset:
    ds = load_dataset("multi_nli", split=split)
    ds = _subsample(ds, sample)
    # unify column names
    ren = {}
    if "sentence1" in ds.column_names:
        ren.update({"sentence1": "premise", "sentence2": "hypothesis"})
    if ren:
        ds = ds.rename_columns(ren)
    return ds


def load_hallucination(split: str, sample: int | None, num_proc: int) -> Dataset:
    """Load WikiBio‑GPT3 hallucination set from HF‑hub or jsonl."""
    # HF id may change; fallback to local jsonl path via --split
    if Path(split).suffix == ".jsonl":
        raw = [json.loads(l) for l in Path(split).open("r", encoding="utf‑8")]
        ds = Dataset.from_list(raw)
    else:
        ds = load_dataset("wiki_bio_gpt3_hallucination", split=split)
    ds = _subsample(ds, sample)

    rename = {}
    for prem_key in ("reference", "wiki", "source"):
        if prem_key in ds.column_names:
            rename[prem_key] = "premise"
            break
    for hyp_key in ("prediction", "candidate", "gpt_sentence", "hypothesis"):
        if hyp_key in ds.column_names and hyp_key != "premise":
            rename[hyp_key] = "hypothesis"
            break
    if rename:
        ds = ds.rename_columns(rename)

    # Ensure integer 0/1 label column named 'label'
    if "label" not in ds.column_names:
        for lab_key in ("is_hallucination", "hallucination", "target"):
            if lab_key in ds.column_names:
                ds = ds.rename_column(lab_key, "label")
                break
    if ds[0]["label"].__class__ is str:
        ds = ds.map(lambda ex: {"label": int(ex["label"])}, num_proc=num_proc)
    return ds


# ---------------------------------------------------------------------------
# Metrics helper (auto‑detect binary vs 3‑way)
# ---------------------------------------------------------------------------


def compute_metrics(preds: List[int], refs: List[int]):
    if len(set(refs)) == 2:  # binary (hallucination)
        bin_preds = [1 if p == LABEL2ID["contradiction"] else 0 for p in preds]
        acc = accuracy_score(refs, bin_preds)
        f1 = f1_score(refs, bin_preds)
        return {"accuracy": acc, "f1": f1}
    acc = accuracy_score(refs, preds)
    f1 = f1_score(refs, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


# ---------------------------------------------------------------------------
# Local HF sequence‑classification inference
# ---------------------------------------------------------------------------


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["premise"], examples["hypothesis"], truncation=True, max_length=256
    )


def eval_local(ds: Dataset, model_name: str, bs: int, num_proc: int):
    tok = AutoTokenizer.from_pretrained(model_name)
    ds_tok = ds.map(
        tokenize_function, batched=True, num_proc=num_proc, fn_kwargs={"tokenizer": tok}
    )
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    collator = DataCollatorWithPadding(tok, return_tensors="pt")
    loader = DataLoader(ds_tok, batch_size=bs, collate_fn=collator)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="HF infer"):
            labels = batch.pop("label")
            logits = model(**{k: v.to(dev) for k, v in batch.items()}).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            refs.extend(labels.cpu().tolist())
    return compute_metrics(preds, refs), preds


# ---------------------------------------------------------------------------
# OpenAI ChatCompletion inference (supports GPT‑4o)
# ---------------------------------------------------------------------------


async def _call_chat(messages, model, retries=3):
    for a in range(retries):
        try:
            resp = await openai_async_client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=200
            )
            return resp.choices[0].message.content
        except Exception as e:
            if a == retries - 1:
                raise e
            await asyncio.sleep(2**a)


def _chunks(seq: Sequence, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def eval_openai(ds: Dataset, model: str, bs: int, max_conn: int, template: str):
    if openai_async_client is None:
        raise RuntimeError("OpenAI SDK not configured.")
    preds, refs = [], []
    sem = asyncio.Semaphore(max_conn)

    async def _worker(prem, hypo, genre):
        async with sem:
            if template == "system_default":
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
                    {
                        "role": "user",
                        "content": f"Premise: {prem}\nHypothesis: {hypo}\nAnswer:",
                    },
                ]
            else:
                prompt = PROMPT_TEMPLATES[template].format(
                    premise=prem, hypothesis=hypo, genre=(genre or "unknown")
                )
                messages = [{"role": "user", "content": prompt}]
            return await _call_chat(messages, model)

    loop = asyncio.get_event_loop()
    for idx_batch in tqdm(list(_chunks(range(len(ds)), bs)), desc="API batches"):
        prem_batch = [ds[i]["premise"] for i in idx_batch]
        hypo_batch = [ds[i]["hypothesis"] for i in idx_batch]
        genre_batch = [ds[i].get("genre", "unknown") for i in idx_batch]
        raw = loop.run_until_complete(
            asyncio.gather(
                *[
                    _worker(p, h, g)
                    for p, h, g in zip(prem_batch, hypo_batch, genre_batch)
                ]
            )
        )
        for r in raw:
            preds.append(
                LABEL2ID.get(parse_openai_response(r, template), LABEL2ID["neutral"])
            )
        refs.extend([ds[i]["label"] for i in idx_batch])
    return compute_metrics(preds, refs), preds


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    # ---------------- dataset ----------------
    if args.task == "nli":
        if args.split.endswith(".jsonl"):
            ds = load_dataset("json", data_files=args.split, split="train")  # type: ignore
            # expect prem/hypo/label already in jsonl
        else:
            ds = load_multinli(args.split, args.sample_size, args.num_proc)
    else:  # hallucination
        ds = load_hallucination(args.split, args.sample_size, args.num_proc)

    print(f"Dataset loaded: {len(ds)} samples  |  columns: {ds.column_names}")

    # ---------------- evaluate ----------------
    if args.model_name.lower().startswith("gpt-"):
        metrics, preds = eval_openai(
            ds,
            args.model_name,
            args.batch_size,
            args.openai_max_conn,
            args.prompt_template,
        )
    else:
        metrics, preds = eval_local(ds, args.model_name, args.batch_size, args.num_proc)

    print("\n==== RESULTS ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if args.save_preds:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        with args.save_preds.open("w", encoding="utf‑8") as f:
            for p, item in zip(preds, ds):
                f.write(
                    f"{item['premise']}\t{item['hypothesis']}\t{ID2LABEL.get(p, p)}\t{item.get('label','?')}\n"
                )
        print(f"Predictions saved to {args.save_preds}")


if __name__ == "__main__":
    main()
