"""Create the stratified 8:1:1 train/validation/test partitions used by KEPPNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_PROCESSED_DIR = Path("data/structured/prad/processed")
DEFAULT_OUTPUT_DIR = Path("data/structured/prad/splits")


def split_data(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 422342,
) -> None:
    """Write non-overlapping, stratified partitions with an approximate 8:1:1 ratio."""
    response_path = Path(processed_dir) / "response_paper.csv"
    response = pd.read_csv(response_path)
    required = {"id", "response"}
    missing = required - set(response.columns)
    if missing:
        raise ValueError(f"{response_path} is missing columns: {sorted(missing)}")
    if response["id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found before partitioning")

    ids = response["id"].astype(str).to_numpy()
    labels = response["response"].astype(int).to_numpy()

    train_val_ids, test_ids, train_val_y, test_y = train_test_split(
        ids,
        labels,
        test_size=0.10,
        stratify=labels,
        random_state=seed,
    )
    train_ids, val_ids, train_y, val_y = train_test_split(
        train_val_ids,
        train_val_y,
        test_size=len(test_ids),
        stratify=train_val_y,
        random_state=seed,
    )

    frames = {
        "training_set.csv": pd.DataFrame({"id": train_ids, "response": train_y}),
        "validation_set.csv": pd.DataFrame({"id": val_ids, "response": val_y}),
        "test_set.csv": pd.DataFrame({"id": test_ids, "response": test_y}),
    }

    split_sets = [set(frame["id"]) for frame in frames.values()]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Partition overlap detected")
    if set.union(*split_sets) != set(ids):
        raise RuntimeError("Partitioning lost or introduced sample IDs")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        frame.to_csv(output_dir / filename, index=False)
    # Dataset configurations use training_split=0.
    frames["training_set.csv"].to_csv(output_dir / "training_set_0.csv", index=False)

    for filename, frame in frames.items():
        counts = frame["response"].value_counts().sort_index().to_dict()
        print(f"{filename}: n={len(frame)}, class_counts={counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=422342)
    args = parser.parse_args()
    split_data(args.processed_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
