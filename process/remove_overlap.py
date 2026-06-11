#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove overlap between a normalized training set and an external test set.
This matches the paper's cross-dataset evaluation practice: normalize external
samples, filter by token length, compute MD5, and remove test samples whose MD5
appears in the training hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Allow running as a standalone script from the project root.
from prepare_divervul_data import (
    dataset_summary,
    load_raw_dataset,
    md5_text,
    normalize_code,
    parse_label,
    read_any_table,
    TokenCounter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove train/test overlap by MD5 of normalized code")
    parser.add_argument("--input_file", required=True, help="External test CSV/JSON/JSONL file")
    parser.add_argument("--train_hashes", required=True, help="Hash file produced by prepare_divervul_data.py")
    parser.add_argument("--code_col", default="code")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--output_file", required=True, help="Filtered CSV output")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tokenizer_name", default=None)
    args = parser.parse_args()

    train_hashes = set(Path(args.train_hashes).read_text(encoding="utf-8").splitlines())
    df = read_any_table(args.input_file)
    if args.code_col not in df.columns or args.label_col not in df.columns:
        raise KeyError(f"Input file must contain {args.code_col!r} and {args.label_col!r}")

    out = df.copy()
    out["label"] = out[args.label_col].apply(parse_label)
    out = out[out["label"].isin([0, 1])].copy()
    out["label"] = out["label"].astype(int)
    out["code"] = out[args.code_col].apply(normalize_code)

    counter = TokenCounter(args.tokenizer_name)
    out["token_count"] = out["code"].apply(counter.count)
    out = out[(out["code"].str.len() > 0) & (out["token_count"] <= args.max_tokens)].copy()
    out["md5"] = out["code"].apply(md5_text)

    before = len(out)
    out = out[~out["md5"].isin(train_hashes)].copy()
    removed = before - len(out)
    out.to_csv(args.output_file, index=False)

    summary = dataset_summary(out)
    summary["overlap_removed"] = removed
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
