"""Lightweight tests for leakage safeguards, splitting, and fusion alignment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate.eval_late_fusion import fuse_and_eval
from scripts.prepare_data.build_patient_summaries import (
    FORBIDDEN_TEXT_MARKERS,
    build_summary_text,
)
from src.data.processor.medical_data.split_data import split_data


class SummarySafetyTest(unittest.TestCase):
    def test_summary_ignores_label_and_source_metadata(self):
        molecular = {
            "mutated_genes": "TP53;PTEN",
            "amp_genes": "MYC",
            "del_genes": "RB1",
            "num_mutated_genes": "2",
            "num_amp_genes": "1",
            "num_del_genes": "1",
            "label": "1",
            "split": "test",
            "study_source": "brca_mbcproject_2022",
            "metastatic_site": "liver",
        }
        text = build_summary_text(
            "breast cancer",
            molecular,
            {"top_pathway_names": "DNA Repair;Cell Cycle"},
        )
        lowered = text.lower()
        for marker in FORBIDDEN_TEXT_MARKERS:
            self.assertNotIn(marker, lowered)
        self.assertIn("TP53", text)
        self.assertIn("DNA Repair", text)


class SplitSafetyTest(unittest.TestCase):
    def test_stratified_8_1_1_split_has_no_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            output = root / "splits"
            processed.mkdir()
            ids = [f"S{i:03d}" for i in range(200)]
            labels = np.array([0, 1] * 100)
            pd.DataFrame({"id": ids, "response": labels}).to_csv(
                processed / "response_paper.csv", index=False
            )

            split_data(processed, output, seed=422342)
            train = pd.read_csv(output / "training_set.csv")
            val = pd.read_csv(output / "validation_set.csv")
            test = pd.read_csv(output / "test_set.csv")

            self.assertEqual((len(train), len(val), len(test)), (160, 20, 20))
            sets = [set(frame["id"]) for frame in (train, val, test)]
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])
            self.assertEqual(set.union(*sets), set(ids))


class FusionAlignmentTest(unittest.TestCase):
    def test_mismatched_branch_labels_are_rejected(self):
        structured = {"S1": (0, 0.2), "S2": (1, 0.8)}
        semantic = {"S1": (1, 0.3), "S2": (1, 0.7)}
        with self.assertRaises(ValueError):
            fuse_and_eval(structured, semantic, 0.5)


if __name__ == "__main__":
    unittest.main()
