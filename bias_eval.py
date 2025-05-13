#!/usr/bin/env python
"""bias_eval.py – Section 3: Social Bias evaluation for CSIT6000R project.

Assesses stereotypical bias in Masked Language Models using two benchmark
corpora:
  1. CrowS‑Pairs  – minimal sentence pairs (stereotype vs. anti‑stereotype)
  2. StereoSet    – intra‑sentence & contextual bias sets (optional)

Metrics implemented
-------------------
* CrowS‑Pairs Stereotype Score  = (# stereotype PLL > anti‑stereotype) / total
* StereoSet  –  SS‑Score (as in original paper) & Language Model Score (LM‑Score)

Usage example
-------------
# Evaluate BERT‑base & RoBERTa‑large on CrowS‑Pairs only
python bias_eval.py \
    --models bert-base-uncased roberta-large \
    --dataset crows \
    --batch-size 32

# Evaluate RoBERTa on both corpora and write CSV results
python bias_eval.py --models roberta-base --dataset both --save-csv results.csv
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Pseudo‑Log‑Likelihood util functions
# ---------------------------------------------------------------------------


def _calc_sentence_pll(model, tokenizer, sentence: str) -> float:
    """Compute pseudo‑log‑likelihood of `sentence` (lower is better)."""
    model.eval()
    with torch.no_grad():
        # Tokenise with special tokens
        input_ids = tokenizer(
            sentence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=128,
        )["input_ids"].to(model.device)
        seq_len = input_ids.size(1)
        pll_sum = 0.0
        # Iterate over tokens except [CLS] / [SEP]
        for i in range(1, seq_len - 1):
            original_token_id = input_ids[0, i].item()
            # Create masked copy
            masked = input_ids.clone()
            masked[0, i] = tokenizer.mask_token_id
            logits = model(masked)[0]
            log_prob = torch.log_softmax(logits[0, i], dim=-1)[original_token_id]
            pll_sum += log_prob.item()
        return pll_sum  # higher is better (log‑probability)


# Batched PLL for speed ------------------------------------------------------


def calc_pll_batched(model, tokenizer, sentences: List[str], batch_size: int = 16):
    """Vectorised PLL – returns list of log‑probs for each sentence."""
    pll_scores = []
    for sent in sentences:
        pll_scores.append(_calc_sentence_pll(model, tokenizer, sent))
    return pll_scores


# ---------------------------------------------------------------------------
# CrowS‑Pairs evaluation
# ---------------------------------------------------------------------------


def eval_crows_pairs(model, tokenizer, batch: int = 16):
    ds = load_dataset("crows_pairs", split="test")
    stereotype_wins = 0
    total = 0
    bias_type_counter = Counter()
    bias_type_wins = Counter()

    for example in tqdm(ds, desc="CrowS"):
        sent_more = example["sent_more"]  # stereotypical
        sent_less = example["sent_less"]  # anti‑stereotypical
        pll_more = _calc_sentence_pll(model, tokenizer, sent_more)
        pll_less = _calc_sentence_pll(model, tokenizer, sent_less)
        total += 1
        bias_type = example["bias_type"]
        bias_type_counter[bias_type] += 1
        if pll_more > pll_less:
            stereotype_wins += 1
            bias_type_wins[bias_type] += 1
    overall_score = stereotype_wins / total
    per_type = {
        bt: bias_type_wins[bt] / count for bt, count in bias_type_counter.items()
    }
    return overall_score, per_type


# ---------------------------------------------------------------------------
# StereoSet evaluation (optional)
# ---------------------------------------------------------------------------


def eval_stereoset(model, tokenizer):
    ds = load_dataset("stereoset", "intrasentence", split="validation")
    ss_correct = 0
    lm_logp_total = 0.0
    total = 0
    for ex in tqdm(ds, desc="StereoSet"):
        target = ex["target_sentence"]
        options = ex["sentences"]["sentence"]
        labels = ex["sentences"][
            "gold_label"
        ]  # 0: unrelated, 1: stereotype, 2: anti‑stereotype
        logps = [_calc_sentence_pll(model, tokenizer, s) for s in options]
        lm_logp_total += sum(logps)
        # choose max logp
        pred_idx = int(torch.tensor(logps).argmax().item())
        gold_idx = labels.index(2)  # anti‑stereotype ideal
        if pred_idx == gold_idx:
            ss_correct += 1
        total += 1
    ss_score = ss_correct / total
    lm_score = lm_logp_total / total
    return ss_score, lm_score


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Bias evaluation for MLMs")
    ap.add_argument("--models", nargs="+", required=True, help="HF model names")
    ap.add_argument(
        "--dataset",
        choices=["crows", "stereoset", "both"],
        default="crows",
        help="Which corpus to evaluate",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save-csv", type=Path)
    return ap.parse_args()


def main():
    args = parse_args()
    results = []

    for model_name in args.models:
        print(f"\n==== Evaluating {model_name} ====")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(args.device)
        row = {"model": model_name}

        if args.dataset in ("crows", "both"):
            score, per_type = eval_crows_pairs(mlm, tokenizer)
            row["crows_overall"] = score
            row.update({f"crows_{k}": v for k, v in per_type.items()})
            print(f"CrowS‑Pairs stereotype ratio: {score:.3f}")

        if args.dataset in ("stereoset", "both"):
            ss_score, lm_score = eval_stereoset(mlm, tokenizer)
            row["ss_score"] = ss_score
            row["lm_score"] = lm_score
            print(f"StereoSet SS‑score: {ss_score:.3f} | LM‑score: {lm_score:.2f}")
        results.append(row)

    # Optional CSV
    if args.save_csv:
        import csv

        fieldnames = sorted({k for r in results for k in r.keys()})
        with args.save_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Results written to {args.save_csv}")


if __name__ == "__main__":
    main()
