#!/usr/bin/env python
"""Encode canonical label-free patient summaries with frozen BioLORD-2023.

The default sample set is ``dual_strict_set`` so the structured and semantic
branches are evaluated on exactly the same patients. Labels and split names are
read only as output metadata; the encoder receives ``summary_text`` alone.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np


LOGGER = logging.getLogger(__name__)


def load_sample_set(cohort_dir: Path, set_name: str) -> List[dict]:
    path = cohort_dir / f"{set_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Sample-set file not found: {path}. Run define_experiment_sets.py first."
        )

    patients: List[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        in_field = f"in_{set_name}"
        if in_field not in reader.fieldnames:
            candidates = [name for name in reader.fieldnames if name.startswith("in_")]
            if len(candidates) != 1:
                raise ValueError(f"Cannot identify inclusion column in {path}")
            in_field = candidates[0]

        required = {"patient_id", "label", "split"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            if str(row.get(in_field, "0")).strip() != "1":
                continue
            patients.append(
                {
                    "patient_id": row["patient_id"].strip(),
                    "label": row["label"].strip(),
                    "split": row["split"].strip(),
                }
            )

    if not patients:
        raise ValueError(f"No included patients found in {path}")
    return patients


def load_summaries(cohort_dir: Path) -> Dict[str, str]:
    path = cohort_dir / "patient_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical summary file not found: {path}. Run build_patient_summaries.py first."
        )

    summaries: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "summary_text"}
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise ValueError(f"{path} must contain patient_id and summary_text")
        for row in reader:
            patient_id = row["patient_id"].strip()
            summary = row["summary_text"].strip()
            if patient_id and summary:
                summaries[patient_id] = summary
    return summaries


def encode_texts(model_id_or_path: str, texts: List[str], device: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required to encode summaries") from exc

    LOGGER.info("Loading frozen BioLORD encoder from %s", model_id_or_path)
    model = SentenceTransformer(model_id_or_path, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancer_type", required=True, choices=["prostate", "breast"])
    parser.add_argument("--biolord_dir", type=Path, default=Path("data/biolord_inputs"))
    parser.add_argument(
        "--model",
        default="FremyCompany/BioLORD-2023",
        help="Hugging Face model ID or a local snapshot path",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--set_name",
        choices=["dual_strict_set", "llm_text_set"],
        default="dual_strict_set",
        help="Patients to encode (default: dual_strict_set)",
    )
    parser.add_argument(
        "--output_prefix",
        default="embeddings",
        help="Output basename; downstream main scripts expect 'embeddings'",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cohort_dir = args.biolord_dir / ("prad" if args.cancer_type == "prostate" else "brca")

    patients = load_sample_set(cohort_dir, args.set_name)
    summaries = load_summaries(cohort_dir)
    missing = [row["patient_id"] for row in patients if row["patient_id"] not in summaries]
    if missing:
        raise ValueError(
            f"{len(missing)} included patients have no non-empty summary; "
            "embedding generation stopped instead of inserting placeholders"
        )

    ordered_ids = [row["patient_id"] for row in patients]
    texts = [summaries[patient_id] for patient_id in ordered_ids]
    embeddings = encode_texts(args.model, texts, args.device, args.batch_size)
    if embeddings.shape[0] != len(ordered_ids):
        raise RuntimeError("Embedding count does not match patient count")

    cohort_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = cohort_dir / f"{args.output_prefix}.npy"
    ids_path = cohort_dir / "embedding_patient_ids.csv"
    meta_path = cohort_dir / "embedding_meta.csv"
    np.save(embedding_path, embeddings)

    with ids_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "patient_id"])
        writer.writerows(enumerate(ordered_ids))

    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["row_index", "patient_id", "label", "split", "cancer_type", "set_name"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(patients):
            writer.writerow(
                {
                    "row_index": index,
                    "patient_id": row["patient_id"],
                    "label": row["label"],
                    "split": row["split"],
                    "cancer_type": args.cancer_type,
                    "set_name": args.set_name,
                }
            )

    LOGGER.info("Saved %s with shape %s", embedding_path, embeddings.shape)
    LOGGER.info("Saved aligned identifiers to %s and metadata to %s", ids_path, meta_path)


if __name__ == "__main__":
    main()
