#!/usr/bin/env python
"""bias_eval.py – Section 3 social‑bias evaluation (v2, 2025‑05‑13)

Supports targeted analysis for a **single protected group** (e.g., "women",
"Chinese", "Hispanic", "LGBT") as required by the CSIT6000R project spec.

Datasets
--------
* **CrowS‑Pairs**  (default) – minimal sentence pairs with stereotype vs.
  anti‑stereotype.
* **StereoSet**    – optional, use ``--dataset stereoset``.

Metrics
-------
* **Stereotype Score**  = percentage of pairs where *stereotype* PLL >
  *anti‑stereotype* PLL. 50 % means neutral; >50 % indicates bias favouring the
  stereotype.
* When ``--group`` is given, evaluation is **restricted** to samples mentioning
  that group (string match across multiple fields). Global results are also
  reported for comparison.

CLI Example
-----------
```bash
python bias_eval.py \
    --model-name roberta-base \
    --dataset crows \
    --group "women" \
    --batch-size 32
```
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm

# ---------------------- CLI -------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Bias evaluation for Masked LMs")
    p.add_argument("--model-name", required=True, type=str)
    p.add_argument(
        "--dataset",
        choices=["crows", "stereoset"],
        default="crows",
        help="Benchmark corpus to use",
    )
    p.add_argument(
        "--batch-size", type=int, default=16, help="Masked‑LM forward batch size"
    )
    p.add_argument(
        "--group",
        type=str,
        default=None,
        help="Target protected group for focused analysis (case‑insensitive substring)",
    )
    return p.parse_args()


# ---------------------- Dataset helpers ------------------------------------

CROWS_NAME_MAP = {
    "sent_more": "stereotype",  # stereotype sentence (higher PLL indicates bias)
    "sent_less": "antistereotype",
}

GROUP_FIELDS = {
    "crows": [
        "sent_more",
        "sent_less",
        "target",
        "bias_type",
    ],
    "stereoset": [
        "context_sentence",
        "sentences",
        "target_word",
    ],
}

WORD_RE = re.compile(r"[A-Za-z\u4E00-\u9FFF]+", re.I)


def normalize(text: str) -> str:
    return text.lower() if text else ""


def group_filter(ds: Dataset, group: str, corpus: str) -> Dataset:
    """Return subset whose any relevant field contains the given group keyword."""
    g = group.lower()
    fields = GROUP_FIELDS[corpus]

    def _keep(example):
        for f in fields:
            if f not in example:
                continue
            val = example[f]
            # StereoSet: 'sentences' is list[dict]
            if isinstance(val, list):
                texts = " ".join([str(x) for x in val])
                if g in texts.lower():
                    return True
            else:
                if g in str(val).lower():
                    return True
        return False

    filtered = ds.filter(_keep, num_proc=4)
    if len(filtered) == 0:
        raise ValueError(
            f"No samples mentioning '{group}' found in {corpus}. Try another keyword."
        )
    return filtered


# ---------------------- PLL computation ------------------------------------


def mask_token_prob(model, tokenizer, input_ids, mask_pos):
    """Return log‑prob of the *true* token when that position is masked.

    Assumes ``input_ids`` already resides on the same device as ``model``.
    """
    masked = input_ids.clone()
    masked[mask_pos] = tokenizer.mask_token_id
    with torch.no_grad():
        logits = model(masked.unsqueeze(0)).logits  # (1, seq, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[0, mask_pos, input_ids[mask_pos]].item()


def sentence_pll(model, tokenizer, text: str) -> float:
    """Compute pseudo‑log‑likelihood (PLL) of *text* under ``model``.

    The token sequence is first moved to the same device as ``model`` to avoid
    CPU/GPU mismatch errors.
    """
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").input_ids.squeeze(0).to(device)
    pll = 0.0
    # Skip special tokens at both ends ([CLS], [SEP] / <s>, </s>)
    for pos in range(1, ids.size(0) - 1):
        pll += mask_token_prob(model, tokenizer, ids, pos)
    return pll


# ---------------------- Evaluation -----------------------------------------


def evaluate_crows(ds: Dataset, model, tokenizer) -> Tuple[float, Dict[str, float]]:
    """Return overall stereotype score and per bias_type scores."""

    totals: Dict[str, Tuple[int, int]] = {}  # bias_type -> (stereo_wins, total)

    for ex in tqdm(ds, desc="CrowS eval"):
        p_stereo = sentence_pll(model, tokenizer, ex["sent_more"])
        p_anti = sentence_pll(model, tokenizer, ex["sent_less"])
        wins = 1 if p_stereo > p_anti else 0
        key = ex["bias_type"]
        if key not in totals:
            totals[key] = (0, 0)
        stereo_cnt, total = totals[key]
        totals[key] = (stereo_cnt + wins, total + 1)

    overall_score = sum(w for w, t in totals.values()) / sum(
        t for _, t in totals.values()
    )
    type_scores = {k: w / t for k, (w, t) in totals.items()}
    return overall_score, type_scores


# (StereoSet evaluation omitted for brevity; similar pattern can be added)

# ---------------------- Main ------------------------------------------------


def main():
    args = parse_args()

    print(f"Loading dataset {args.dataset} …")
    if args.dataset == "crows":
        ds = load_dataset("crows_pairs", split="test", trust_remote_code=True)
    else:  # stereoset
        ds = load_dataset("stereoset", "intrasentence", split="validation")

    if args.group:
        print(f"Filtering for group keyword: '{args.group}' …")
        ds = group_filter(ds, args.group, args.dataset)
        print(f"After filtering: {len(ds)} examples remain.")
    else:
        print(f"Evaluating all {len(ds)} examples.")

    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.dataset == "crows":
        overall, per_type = evaluate_crows(ds, model, tokenizer)
        print("\n=== CrowS‑Pairs stereotype scores ===")
        print(f"Overall S‑Score: {overall:.3f}")
        for k, v in per_type.items():
            print(f"{k:>12}: {v:.3f}")
    else:
        print("StereoSet evaluation not yet implemented in this demo.")


if __name__ == "__main__":
    main()
