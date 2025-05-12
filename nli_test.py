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
# Prompt Templates
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "1_direct_instruction": """You are an expert natural-language-inference assistant.
Task: Given a PREMISE and a HYPOTHESIS, decide whether the hypothesis is (A) ENTAILED, (B) NEUTRAL, or (C) CONTRADICTED by the premise.
Return exactly one word: "entailment", "neutral", or "contradiction".

Premise: {premise}
Hypothesis: {hypothesis}
Answer:""",

    "2_chain_of_thought": """You are a logical reasoner.
For each input, think step by step.
First output your reasoning under the tag "Reasoning:", then output "Answer:" followed by exactly one of [entailment | neutral | contradiction].

Premise: {premise}
Hypothesis: {hypothesis}

Reasoning:""",

    "3_summarize_compare_label": """Step 1 - Premise summary: ___
Step 2 - Hypothesis summary: ___
Step 3 - If premise truthfully guarantees hypothesis → entailment; if contradicts → contradiction; else → neutral.
Output result in the form: Answer: <label>

Premise: {premise}
Hypothesis: {hypothesis}""",

    "6_genre_aware_few_shot": """Task: Natural Language Inference
Definition: Decide whether H is ENTAILED by, CONTRADICTS, or is NEUTRAL to P.

Example 1
Genre: LETTERS
P: The garden scheme teaches children the value of the land.
H: All children love gardening.
Label: contradiction

Example 2
Genre: 9/11
P: At 8:34, the Boston Center controller received a third transmission from American 11.
H: The Boston Center controller got a third transmission from American 11.
Label: entailment

Now solve:
Genre: {genre}
P: {premise}
H: {hypothesis}
Label:"""
}

# Default System Prompt (original behavior)
SYSTEM_PROMPT_DEFAULT = (
    "You are an NLI classifier. Given a premise and a hypothesis, "
    "respond with one of exactly three words: entailment, neutral, or contradiction."
)


def parse_openai_response(response_text: str, prompt_template_key: str) -> str:
    """Parses the model's response to extract the label."""
    text = response_text.strip().lower()
    label = ""

    if prompt_template_key == "2_chain_of_thought" or prompt_template_key == "3_summarize_compare_label":
        # Expect "Reasoning: ... Answer: <label>" or "Answer: <label>"
        parts = text.split("answer:")
        if len(parts) > 1:
            # Take first word after the last "answer:"
            label = parts[-1].strip().split()[0] if parts[-1].strip() else ""
        else: # Fallback: assume the whole response is the label or the relevant part
            label = text.split()[0] if text else ""
    elif prompt_template_key == "6_genre_aware_few_shot":
        # Expect "Label: <label>" or just "<label>"
        parts = text.split("label:")
        if len(parts) > 1:
            label = parts[-1].strip().split()[0] if parts[-1].strip() else ""
        else: # Fallback
            label = text.split()[0] if text else ""
    elif prompt_template_key == "1_direct_instruction" or prompt_template_key == "system_default":
        # Expect just the label word
        label = text.split()[0] if text else ""
    else: # Should not happen if prompt_template_key is validated
        label = text.split()[0] if text else ""
    
    return label.rstrip(".,") # Clean trailing punctuation


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
    parser.add_argument("--split", type=str, default="dev_matched", 
                        help="Dataset split (e.g., dev_matched, dev_mismatched) or path to a .jsonl file.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per‑device batch size")
    parser.add_argument("--sample-size", type=int, default=None, help="Subset for quick tests")
    parser.add_argument("--num-proc", type=int, default=4, help="HF map() workers")
    parser.add_argument("--openai-max-conn", type=int, default=5, help="并发请求数")
    parser.add_argument("--save-preds-path", type=Path, default=None, help="Path to save predictions (e.g., preds.tsv)")
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="system_default",
        choices=["system_default"] + list(PROMPT_TEMPLATES.keys()),
        help="Prompt template to use for OpenAI models. 'system_default' uses the original hardcoded prompt."
    )
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

def load_jsonl_data(file_path: Path, sample_size: int | None) -> "datasets.Dataset":
    """Load a dataset from a JSONL file."""
    data = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        raise ValueError(f"No data found in {file_path}")

    # Convert to Hugging Face Dataset
    # We'll create a dictionary of lists for Dataset.from_dict
    processed_data = {key: [] for key in data[0].keys()}
    # Add a new key for the integer label
    if "gold_label" in data[0] and "label" not in data[0]: # Ensure 'label' isn't already a key
        processed_data["label"] = []


    for item in data:
        for key, value in item.items():
            if key == "gold_label": # Map gold_label to integer label
                 processed_data["label"].append(LABEL2ID.get(str(value).lower(), -1)) # -1 for unknown/missing
                 # Also keep original gold_label if needed, or just map to 'label'
                 if "gold_label" in processed_data: # if gold_label is an original key
                    processed_data["gold_label"].append(value)
            elif key in processed_data:
                processed_data[key].append(value)
            else: # New key encountered after first row
                processed_data[key] = [None] * (len(processed_data[next(iter(processed_data))]) -1) + [value]


    ds = datasets.Dataset.from_dict(processed_data)

    if sample_size and sample_size < len(ds):
        ds = ds.shuffle(seed=42).select(range(sample_size))

    # 统一列名：premise / hypothesis / label
    rename_map = {}
    if "sentence1" in ds.column_names and "premise" not in ds.column_names:
        rename_map["sentence1"] = "premise"
    if "sentence2" in ds.column_names and "hypothesis" not in ds.column_names:
        rename_map["sentence2"] = "hypothesis"
    
    # If 'gold_label' was the source for 'label' and we don't need 'gold_label' anymore,
    # it could be removed here, but it's often useful to keep.
    # The 'label' column should now contain integer IDs.

    if rename_map:
        ds = ds.rename_columns(rename_map)
    
    # Ensure 'label' column exists if 'gold_label' was present
    if "gold_label" in data[0] and "label" not in ds.column_names and "label" in processed_data:
        # This case should be handled by the initial creation of processed_data["label"]
        # and Dataset.from_dict. If 'label' is still missing, there's an issue.
        pass
    elif "label" not in ds.column_names:
        # If there's no 'gold_label' and no 'label', we might not have labels.
        # This is fine for prediction-only tasks, but metrics will fail.
        print("Warning: 'label' column not found in the JSONL data. Metrics requiring labels will not be computed.")


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
    # Ensure 'label' is present for refs, even if not used by model directly for prediction
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

    all_preds, all_refs = [], []
    for batch in tqdm(dataloader, desc="Inference"):
        # Extract labels for reference before moving to device if they exist
        labels = batch.pop("label", None) # Use pop to remove label from batch sent to model
        
        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch_on_device).logits
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        if labels is not None:
            all_refs.extend(labels.cpu().tolist())
        else: # Fallback if labels are not in the batch for some reason (should not happen with current ds_tok.set_format)
            # This branch might indicate an issue if 'label' is expected but not found.
            # For now, we'll assume this means we can't compute metrics that need refs for this batch.
            # Or, if ds doesn't have 'label' (e.g. test set without labels), refs will be empty.
            pass


    metrics = compute_metrics(all_preds, all_refs) if all_refs else {"accuracy": float('nan'), "macro_f1": float('nan')}
    return metrics, all_preds


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


async def predict_openai_async(
    premises: List[str], 
    hypotheses: List[str], 
    genres: List[str | None], 
    model: str, 
    max_conn: int, 
    prompt_template_key: str
):
    """并发请求 OpenAI ChatCompletion，返回字符串标签。"""
    semaphore = asyncio.Semaphore(max_conn)

    async def _worker(premise: str, hypothesis: str, genre: str | None):
        async with semaphore:
            messages: List[dict]
            if prompt_template_key == "system_default":
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
                    {
                        "role": "user",
                        "content": (
                            f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer with one word."
                        ),
                    },
                ]
            elif prompt_template_key in PROMPT_TEMPLATES:
                prompt_content = PROMPT_TEMPLATES[prompt_template_key]
                # Provide a default for genre, format() will ignore it if {genre} is not in prompt_content
                current_genre = genre if genre else "unknown" 
                formatted_prompt = prompt_content.format(premise=premise, hypothesis=hypothesis, genre=current_genre)
                messages = [{"role": "user", "content": formatted_prompt}]
            else:
                # Should not be reached if arg parsing validates choices
                raise ValueError(f"Unknown prompt template key: {prompt_template_key}")
            
            return await _call_chat_api(messages, model=model)

    tasks = [_worker(p, h, g) for p, h, g in zip(premises, hypotheses, genres)]
    return await asyncio.gather(*tasks)


def eval_openai(ds, model_name: str, batch_size: int, max_conn: int, prompt_template_key: str):
    """Batch evaluate OpenAI model via ChatCompletion API."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")

    all_preds, all_refs = [], []
    for batch_indices in tqdm(list(chunk(range(len(ds)), batch_size)), desc="API batches"):
        prem_batch = [ds[i]["premise"] for i in batch_indices]
        hyp_batch = [ds[i]["hypothesis"] for i in batch_indices]
        
        # Ensure 'label' column exists for references
        if "label" not in ds.column_names:
            raise ValueError("Dataset is missing the 'label' column, which is required for evaluation references.")
        labels_batch = [ds[i]["label"] for i in batch_indices]


        genres_batch: List[str | None] = [None] * len(prem_batch)
        if prompt_template_key == "6_genre_aware_few_shot":
            if "genre" not in ds.column_names:
                # This check is also in main, but good to be defensive
                print("Warning: 'genre' column not found in dataset for all items. Prompt 6 may not work as expected.")
                # Fallback to "unknown" if genre is missing for an item
                genres_batch = [ds[i].get("genre", "unknown") for i in batch_indices]
            else:
                genres_batch = [ds[i]["genre"] for i in batch_indices]
        
        # Fallback for older asyncio versions if get_event_loop is deprecated and new_event_loop/set_event_loop is preferred
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError: # pragma: no cover
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)


        raw_outputs = loop.run_until_complete(
            predict_openai_async(prem_batch, hyp_batch, genres_batch, model=model_name, max_conn=max_conn, prompt_template_key=prompt_template_key)
        )

        # Map string answers back to id (default to neutral = 1 if unknown)
        current_preds = []
        for o_text in raw_outputs:
            parsed_label_str = parse_openai_response(str(o_text), prompt_template_key)
            current_preds.append(LABEL2ID.get(parsed_label_str, 1)) # Default to neutral (1)
        
        all_preds.extend(current_preds)
        all_refs.extend(labels_batch)

    metrics = compute_metrics(all_preds, all_refs) if all_refs else {"accuracy": float('nan'), "macro_f1": float('nan')}
    return metrics, all_preds

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

    if args.split.endswith(".jsonl"):
        print(f"Loading data from JSONL file: {args.split}")
        ds = load_jsonl_data(Path(args.split), args.sample_size)
    else:
        print(f"Loading data from Hugging Face Hub: multi_nli, split: {args.split}")
        ds = load_multinli(args.split, args.sample_size, args.num_proc)
    
    # Ensure 'label' column exists if we are going to compute metrics
    if "label" not in ds.column_names:
        print("Warning: 'label' column not found in the dataset. Metrics cannot be computed.")
        # Depending on strictness, you might want to raise an error or proceed without metrics.

    # Specific check for prompt 6 and genre
    if args.model_name.lower().startswith("gpt-") and args.prompt_template == "6_genre_aware_few_shot":
        if "genre" not in ds.column_names:
            raise ValueError("Prompt template '6_genre_aware_few_shot' selected, but 'genre' column is missing from the loaded dataset.")

    predictions = []
    if args.model_name.lower().startswith("gpt-"):
        print(f"Using OpenAI model: {args.model_name} with prompt template: {args.prompt_template}")
        res, predictions = eval_openai(ds, args.model_name, args.batch_size, args.openai_max_conn, args.prompt_template)
    else:
        res, predictions = eval_local_model(ds, args.model_name, args.batch_size, args.num_proc)

    # 拓展：可保存预测，或写入 CSV 供后续分析
    print("=== Evaluation Results ===")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")

    if args.save_preds_path and predictions:
        print(f"Saving predictions to {args.save_preds_path}...")
        save_preds(predictions, ds, args.save_preds_path)
        print("Predictions saved.")



if __name__ == "__main__":
    main()
