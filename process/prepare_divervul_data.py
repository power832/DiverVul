#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiverVul data preprocessing pipeline.

Pipeline:
1) Read raw function-level vulnerability data from CSV / JSON / JSONL.
2) Normalize code: remove C/C++ comments, blank lines, common escape characters,
   and standardize line breaks while preserving original labels.
3) Filter samples longer than a max token threshold.
4) Deduplicate normalized code by MD5.
5) Optionally sample a manageable subset with a fixed random seed.
6) Split data into train / valid / test with stratification.
7) Pair each code sample with exactly one instruction from the instruction pool.
8) Export Alpaca-style JSONL files: {instruction, input, output}.

Example:
python src/prepare_divervul_data.py \
  --input_files examples/raw_sample.csv \
  --code_col code \
  --label_col label \
  --instruction_file instructions/instruction_pool.txt \
  --output_dir data/processed \
  --max_tokens 512 \
  --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Stats:
    raw: int = 0
    missing_removed: int = 0
    invalid_label_removed: int = 0
    length_removed: int = 0
    duplicate_removed: int = 0
    final: int = 0
    vulnerable: int = 0
    non_vulnerable: int = 0


def read_any_table(path: str | Path) -> pd.DataFrame:
    """Read CSV, JSON, or JSONL into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".jl"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}. Use CSV, JSON, or JSONL.")


def parse_label(value) -> Optional[int]:
    """Convert common vulnerability labels to 0/1."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in {0, 1} else None
    if isinstance(value, float):
        if value in {0.0, 1.0}:
            return int(value)
        return None

    text = str(value).strip().lower()
    positive = {"1", "true", "yes", "y", "vul", "vulnerable", "bad", "unsafe"}
    negative = {"0", "false", "no", "n", "non-vul", "non_vul", "non-vulnerable", "safe", "clean"}
    if text in positive:
        return 1
    if text in negative:
        return 0
    return None


def unescape_common_chars(code: str) -> str:
    """Handle common escaped line breaks and tabs found in CSV/JSON dumps."""
    code = str(code)
    replacements = {
        "\\r\\n": "\n",
        "\\n": "\n",
        "\\t": "\t",
        "\r\n": "\n",
        "\r": "\n",
        "\u000d\u000a": "\n",
    }
    for src, tgt in replacements.items():
        code = code.replace(src, tgt)
    return code


def remove_c_cpp_comments(code: str) -> str:
    """
    Remove C/C++ line and block comments with a small state machine.
    This avoids deleting // or /* */ that appear inside strings/chars.
    """
    NORMAL, LINE_COMMENT, BLOCK_COMMENT, STRING, CHAR = range(5)
    state = NORMAL
    out: List[str] = []
    i = 0
    n = len(code)

    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""

        if state == NORMAL:
            if c == "/" and nxt == "/":
                state = LINE_COMMENT
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = BLOCK_COMMENT
                i += 2
                continue
            if c == '"':
                state = STRING
                out.append(c)
                i += 1
                continue
            if c == "'":
                state = CHAR
                out.append(c)
                i += 1
                continue
            out.append(c)
            i += 1
            continue

        if state == LINE_COMMENT:
            if c == "\n":
                out.append("\n")
                state = NORMAL
            i += 1
            continue

        if state == BLOCK_COMMENT:
            if c == "*" and nxt == "/":
                state = NORMAL
                i += 2
            else:
                if c == "\n":
                    out.append("\n")
                i += 1
            continue

        if state == STRING:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
                continue
            if c == '"':
                state = NORMAL
            i += 1
            continue

        if state == CHAR:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
                continue
            if c == "'":
                state = NORMAL
            i += 1
            continue

    return "".join(out)


def normalize_code(code: str) -> str:
    """
    Conservative function-level code normalization.
    It removes comments and blank lines, normalizes line breaks and tabs,
    and keeps the main code statements unchanged.
    """
    code = unescape_common_chars(code)
    code = remove_c_cpp_comments(code)
    code = code.replace("\t", "    ").replace("\x00", "")

    normalized_lines = []
    for line in code.split("\n"):
        line = line.rstrip()
        line = re.sub(r"[ \f\v]+", " ", line).strip()
        if line:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


class TokenCounter:
    """Token counter using a Hugging Face tokenizer when available; otherwise regex fallback."""

    def __init__(self, tokenizer_name: Optional[str] = None):
        self.tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load tokenizer {tokenizer_name!r}. "
                    "Install transformers and ensure the model/tokenizer is available."
                ) from exc

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        # Fallback: approximate code tokens by identifiers, numbers, and punctuation.
        return len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]", text))


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def load_raw_dataset(input_files: Iterable[str], code_col: str, label_col: str) -> Tuple[pd.DataFrame, Stats]:
    frames = []
    for file in input_files:
        df = read_any_table(file)
        if code_col not in df.columns or label_col not in df.columns:
            raise KeyError(
                f"{file} must contain columns {code_col!r} and {label_col!r}. "
                f"Available columns: {list(df.columns)}"
            )
        df = df.copy()
        df["source_file"] = Path(file).name
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    stats = Stats(raw=len(raw))
    return raw, stats


def preprocess_dataframe(
    df: pd.DataFrame,
    code_col: str,
    label_col: str,
    max_tokens: int,
    tokenizer_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Stats]:
    stats = Stats(raw=len(df))
    work = df[[code_col, label_col] + [c for c in df.columns if c not in {code_col, label_col}]].copy()
    work = work.rename(columns={code_col: "raw_code", label_col: "raw_label"})

    before = len(work)
    work = work.dropna(subset=["raw_code", "raw_label"])
    stats.missing_removed = before - len(work)

    work["label"] = work["raw_label"].apply(parse_label)
    before = len(work)
    work = work[work["label"].isin([0, 1])].copy()
    stats.invalid_label_removed = before - len(work)
    work["label"] = work["label"].astype(int)

    work["code"] = work["raw_code"].apply(normalize_code)
    before = len(work)
    work = work[work["code"].str.len() > 0].copy()
    stats.missing_removed += before - len(work)

    counter = TokenCounter(tokenizer_name)
    work["token_count"] = work["code"].apply(counter.count)
    before = len(work)
    work = work[work["token_count"] <= max_tokens].copy()
    stats.length_removed = before - len(work)

    work["md5"] = work["code"].apply(md5_text)
    before = len(work)
    work = work.drop_duplicates(subset=["md5"], keep="first").copy()
    stats.duplicate_removed = before - len(work)

    work = work.reset_index(drop=True)
    stats.final = len(work)
    stats.vulnerable = int((work["label"] == 1).sum())
    stats.non_vulnerable = int((work["label"] == 0).sum())
    return work, stats


def stratified_sample(
    df: pd.DataFrame,
    sample_total: Optional[int],
    sample_vul: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Optional reproducible subset sampling."""
    if sample_total is None:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if sample_total > len(df):
        sample_total = len(df)

    rng = np.random.default_rng(seed)
    vul_df = df[df["label"] == 1]
    non_df = df[df["label"] == 0]

    if sample_vul is not None:
        n_vul = min(sample_vul, len(vul_df), sample_total)
        n_non = min(sample_total - n_vul, len(non_df))
    else:
        # Preserve the current class ratio when no target vulnerable count is given.
        ratio = len(vul_df) / max(len(df), 1)
        n_vul = min(int(round(sample_total * ratio)), len(vul_df))
        n_non = min(sample_total - n_vul, len(non_df))

    sampled = pd.concat(
        [
            vul_df.sample(n=n_vul, random_state=seed) if n_vul > 0 else vul_df.iloc[:0],
            non_df.sample(n=n_non, random_state=seed + 1) if n_non > 0 else non_df.iloc[:0],
        ],
        ignore_index=True,
    )
    # If one class is insufficient, fill remaining slots from the other class.
    remaining = sample_total - len(sampled)
    if remaining > 0:
        used_md5 = set(sampled["md5"])
        rest = df[~df["md5"].isin(used_md5)]
        if len(rest) > 0:
            sampled = pd.concat(
                [sampled, rest.sample(n=min(remaining, len(rest)), random_state=seed + 2)],
                ignore_index=True,
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each label group into train/valid/test using fixed seed."""
    parts = {"train": [], "valid": [], "test": []}
    for _, group in df.groupby("label"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_valid = int(round(n * valid_ratio))
        # Keep all remaining samples in test to avoid losing rows.
        parts["train"].append(group.iloc[:n_train])
        parts["valid"].append(group.iloc[n_train : n_train + n_valid])
        parts["test"].append(group.iloc[n_train + n_valid :])

    train = pd.concat(parts["train"], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid = pd.concat(parts["valid"], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(parts["test"], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, valid, test


def load_instructions(path: str | Path) -> List[str]:
    path = Path(path)
    instructions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                instructions.append(line)
    if not instructions:
        raise ValueError(f"No instructions found in {path}")
    return instructions


def pair_and_format(df: pd.DataFrame, instructions: List[str], seed: int) -> List[dict]:
    rng = random.Random(seed)
    records = []
    for _, row in df.iterrows():
        instruction = rng.choice(instructions)
        records.append(
            {
                "instruction": instruction,
                "input": row["code"],
                "output": str(int(row["label"])),
            }
        )
    return records


def write_jsonl(records: List[dict], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_hashes(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for h in df["md5"].tolist():
            f.write(str(h) + "\n")


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "samples": int(len(df)),
        "vulnerable": int((df["label"] == 1).sum()),
        "non_vulnerable": int((df["label"] == 0).sum()),
        "avg_tokens": float(df["token_count"].mean()) if len(df) else 0.0,
        "max_tokens": int(df["token_count"].max()) if len(df) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DiverVul preprocessing and Alpaca SFT data builder")
    parser.add_argument("--input_files", nargs="+", required=True, help="Raw CSV/JSON/JSONL files")
    parser.add_argument("--code_col", default="code", help="Column containing function-level code")
    parser.add_argument("--label_col", default="label", help="Column containing 0/1 vulnerability label")
    parser.add_argument("--instruction_file", default="instructions/instruction_pool.txt")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tokenizer_name", default=None, help="Optional HF tokenizer, e.g., codellama/CodeLlama-13b-hf")
    parser.add_argument("--sample_total", type=int, default=None, help="Optional final sample size, e.g., 25915")
    parser.add_argument("--sample_vul", type=int, default=None, help="Optional vulnerable count inside sampled subset, e.g., 8931")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, load_stats = load_raw_dataset(args.input_files, args.code_col, args.label_col)
    clean, stats = preprocess_dataframe(
        raw,
        code_col=args.code_col,
        label_col=args.label_col,
        max_tokens=args.max_tokens,
        tokenizer_name=args.tokenizer_name,
    )

    clean = stratified_sample(clean, args.sample_total, args.sample_vul, args.seed)
    train, valid, test = stratified_split(clean, args.train_ratio, args.valid_ratio, args.seed)

    instructions = load_instructions(args.instruction_file)
    train_records = pair_and_format(train, instructions, args.seed)
    valid_records = pair_and_format(valid, instructions, args.seed + 1)
    test_records = pair_and_format(test, instructions, args.seed + 2)

    clean.to_csv(output_dir / "clean_samples.csv", index=False)
    train.to_csv(output_dir / "train_clean.csv", index=False)
    valid.to_csv(output_dir / "valid_clean.csv", index=False)
    test.to_csv(output_dir / "test_clean.csv", index=False)

    write_jsonl(train_records, output_dir / "sft_train.jsonl")
    write_jsonl(valid_records, output_dir / "sft_valid.jsonl")
    write_jsonl(test_records, output_dir / "sft_test.jsonl")

    write_hashes(train, output_dir / "train_hashes.txt")
    write_hashes(clean, output_dir / "all_clean_hashes.txt")

    summary = {
        "preprocess_stats": stats.__dict__,
        "sampled_clean": dataset_summary(clean),
        "train": dataset_summary(train),
        "valid": dataset_summary(valid),
        "test": dataset_summary(test),
        "instruction_pool_size": len(instructions),
        "config": vars(args),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDone. Alpaca SFT files are saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
