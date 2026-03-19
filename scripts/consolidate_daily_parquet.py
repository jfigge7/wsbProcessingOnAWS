#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def iter_day_dirs(root: Path, dataset: str) -> Iterable[Path]:
    yield from sorted(root.joinpath(dataset).glob("year=*/month=*/day=*"))


def parquet_files(day_dir: Path) -> List[Path]:
    return sorted(p for p in day_dir.glob("*.parquet") if p.is_file())


def merge_day(day_dir: Path, *, output_name: str, delete_inputs: bool, dry_run: bool) -> dict:
    files = parquet_files(day_dir)
    if len(files) <= 1:
        return {"day_dir": str(day_dir), "status": "skipped", "inputs": len(files)}

    output_path = day_dir / output_name
    if output_path in files:
        source_files = [p for p in files if p != output_path]
    else:
        source_files = files

    if not dry_run:
        frames = [pd.read_parquet(p) for p in source_files]
        merged = pd.concat(frames, ignore_index=True)
        tmp_path = day_dir / f".{output_name}.tmp"
        merged.to_parquet(tmp_path, index=False)
        tmp_path.replace(output_path)

        if delete_inputs:
            for path in files:
                if path != output_path:
                    path.unlink()

    return {
        "day_dir": str(day_dir),
        "status": "merged",
        "inputs": len(files),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collapse partitioned daily parquet shards into one parquet file per day."
    )
    parser.add_argument(
        "--root",
        default="comments_and_submissions",
        help="Root directory containing comments/ and submissions/ partitions.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["comments", "submissions"],
        default=["comments", "submissions"],
        help="Datasets to consolidate.",
    )
    parser.add_argument(
        "--output-name",
        default="data_0.parquet",
        help="Filename to use for the merged daily parquet.",
    )
    parser.add_argument(
        "--keep-inputs",
        action="store_true",
        help="Keep original shard files instead of deleting them after merge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be merged without modifying files.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    total_merged = 0
    total_skipped = 0

    for dataset in args.datasets:
        dataset_merged = 0
        dataset_skipped = 0
        for day_dir in iter_day_dirs(root, dataset):
            result = merge_day(
                day_dir,
                output_name=args.output_name,
                delete_inputs=not args.keep_inputs,
                dry_run=args.dry_run,
            )
            if result["status"] == "merged":
                dataset_merged += 1
            else:
                dataset_skipped += 1

        total_merged += dataset_merged
        total_skipped += dataset_skipped
        print(
            f"{dataset}: merged_days={dataset_merged} skipped_days={dataset_skipped} dry_run={args.dry_run}"
        )

    print(
        f"total: merged_days={total_merged} skipped_days={total_skipped} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
