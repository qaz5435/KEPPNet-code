#!/usr/bin/env python
"""Build deterministic, label-free patient summaries for BioLORD.

The text encoder is allowed to see molecular information only: somatic
mutations, copy-number amplifications/deletions, alteration counts, and
pathway involvement derived from those alterations. Clinical outcome labels,
sample status, anatomical site, study/source identifiers, and split names are
never inserted into ``summary_text``.

Expected inputs for each cohort directory under ``data/biolord_inputs``:

* ``patient_master.csv`` with at least ``patient_id`` (other columns are ignored)
* ``patient_mutated_genes.csv`` with mutation/CNA lists and counts
* ``patient_top_pathways.csv`` with molecularly derived pathway names

The canonical output is ``patient_summary.csv``. It intentionally contains no
label or split column; labels and partitions are joined later by
``define_experiment_sets.py``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List


LOGGER = logging.getLogger(__name__)

CONFIGS = {
    "prostate": {"subdir": "prad", "cancer_display": "prostate cancer"},
    "breast": {"subdir": "brca", "cancer_display": "breast cancer"},
}

# These markers are forbidden in the encoded text because they can reveal the
# target directly or act as cohort/status proxies.
FORBIDDEN_TEXT_MARKERS = (
    "primary",
    "metastatic",
    "metastasis",
    "tumor site",
    "data source",
    "study source",
    "training set",
    "validation set",
    "test set",
    "brca_",
    "prad_",
)


def _read_csv_by_id(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    rows: Dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "patient_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a patient_id column")
        for row in reader:
            patient_id = str(row.get("patient_id", "")).strip()
            if patient_id:
                rows[patient_id] = row
    return rows


def _split_items(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def _format_items(value: str, maximum: int, empty_text: str) -> str:
    items = _split_items(value)
    if not items:
        return empty_text
    shown = items[:maximum]
    suffix = f" and {len(items) - maximum} others" if len(items) > maximum else ""
    return ", ".join(shown) + suffix


def _nonnegative_int(row: dict, key: str, fallback: int) -> int:
    raw = row.get(key, "")
    if raw in (None, ""):
        return fallback
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {key}: {raw!r}") from exc


def build_summary_text(cancer_display: str, molecular_row: dict, pathway_row: dict) -> str:
    """Construct one standardized summary using molecular fields only."""
    mutated_genes = _split_items(molecular_row.get("mutated_genes", ""))
    amplified_genes = _split_items(molecular_row.get("amp_genes", ""))
    deleted_genes = _split_items(molecular_row.get("del_genes", ""))

    n_mut = _nonnegative_int(molecular_row, "num_mutated_genes", len(mutated_genes))
    n_amp = _nonnegative_int(molecular_row, "num_amp_genes", len(amplified_genes))
    n_del = _nonnegative_int(molecular_row, "num_del_genes", len(deleted_genes))
    n_total = n_mut + n_amp + n_del

    sentences = [f"A {cancer_display} patient."]

    if mutated_genes:
        genes = _format_items(molecular_row.get("mutated_genes", ""), 15, "none")
        sentences.append(f"Somatic mutations affect {n_mut} genes, including {genes}.")
    else:
        sentences.append("No somatic mutation events were recorded in the selected gene set.")

    cna_parts: List[str] = []
    if amplified_genes:
        genes = _format_items(molecular_row.get("amp_genes", ""), 10, "none")
        cna_parts.append(f"amplification affects {n_amp} genes, including {genes}")
    if deleted_genes:
        genes = _format_items(molecular_row.get("del_genes", ""), 10, "none")
        cna_parts.append(f"deletion affects {n_del} genes, including {genes}")
    if cna_parts:
        sentences.append("Copy-number " + "; ".join(cna_parts) + ".")
    else:
        sentences.append("No copy-number amplification or deletion events were recorded in the selected gene set.")

    sentences.append(f"The molecular profile contains {n_total} recorded alteration events across channels.")

    pathway_names = pathway_row.get("top_pathway_names", pathway_row.get("top_pathways", ""))
    pathways = _format_items(
        pathway_names,
        8,
        "no pathway-level involvement identified from the recorded alterations",
    )
    sentences.append(f"The most involved biological pathways include {pathways}.")

    text = " ".join(sentences)
    lowered = text.lower()
    hits = [marker for marker in FORBIDDEN_TEXT_MARKERS if marker in lowered]
    if hits:
        raise ValueError(f"Forbidden target/proxy marker(s) in generated summary: {hits}")
    return text


def process(cancer_type: str, input_root: Path) -> Path:
    cfg = CONFIGS[cancer_type]
    cohort_dir = input_root / cfg["subdir"]
    master = _read_csv_by_id(cohort_dir / "patient_master.csv")
    molecular = _read_csv_by_id(cohort_dir / "patient_mutated_genes.csv")
    pathways = _read_csv_by_id(cohort_dir / "patient_top_pathways.csv")

    missing_molecular = sorted(set(master) - set(molecular))
    missing_pathways = sorted(set(master) - set(pathways))
    if missing_molecular or missing_pathways:
        details = []
        if missing_molecular:
            details.append(f"{len(missing_molecular)} missing molecular rows")
        if missing_pathways:
            details.append(f"{len(missing_pathways)} missing pathway rows")
        raise ValueError(
            f"Incomplete semantic inputs for {cancer_type}: " + ", ".join(details)
        )

    output = cohort_dir / "patient_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patient_id", "summary_text", "cancer_type"],
        )
        writer.writeheader()
        for patient_id in sorted(master):
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "summary_text": build_summary_text(
                        cfg["cancer_display"],
                        molecular[patient_id],
                        pathways[patient_id],
                    ),
                    "cancer_type": cancer_type,
                }
            )

    LOGGER.info("Wrote %d label-free summaries to %s", len(master), output)
    return output


def _selected_cancers(value: str) -> Iterable[str]:
    return CONFIGS if value == "all" else (value,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cancer_type",
        choices=["prostate", "breast", "all"],
        default="all",
        help="Cohort to process (default: all)",
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("data/biolord_inputs"),
        help="Root containing prad/ and brca/ semantic-input directories",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for cancer_type in _selected_cancers(args.cancer_type):
        process(cancer_type, args.input_root)


if __name__ == "__main__":
    main()
