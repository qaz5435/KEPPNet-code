#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_demo.py — KEPPNet minimal demo using example_data/.

This script runs a self-contained forward-pass demo using the synthetic
example data in example_data/. It does NOT require real patient data,
Reactome files, or STRING PPI data.

What it demonstrates:
  1. Load synthetic mutation / CNA / response data from example_data/
  2. Build a minimal dense network (no biological mask) as a structural proxy
  3. Run a forward pass and compute basic metrics
  4. Show the expected output format for attribution analysis

For a full run with real data, see docs/data_preparation.md.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --cancer_type prostate
    python scripts/run_demo.py --verbose
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "example_data")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_example_data():
    """Load synthetic example data from example_data/."""
    mut_path  = os.path.join(EXAMPLE_DIR, "sample_mutation.csv")
    cna_path  = os.path.join(EXAMPLE_DIR, "sample_cna.csv")
    resp_path = os.path.join(EXAMPLE_DIR, "sample_response.csv")

    for p in [mut_path, cna_path, resp_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Example data not found: {p}\n"
                "Make sure you are running from the KEPPNET repository root."
            )

    mut  = pd.read_csv(mut_path,  index_col=0)
    cna  = pd.read_csv(cna_path,  index_col=0)
    resp = pd.read_csv(resp_path, index_col=0)

    # Align samples
    common = mut.index.intersection(cna.index).intersection(resp.index)
    mut, cna, resp = mut.loc[common], cna.loc[common], resp.loc[common]

    # Build combined feature matrix: mut | cna_amp | cna_del
    genes = mut.columns.intersection(cna.columns)
    cna_amp = cna[genes].clip(lower=0)
    cna_del = (-cna[genes]).clip(lower=0)

    X = np.concatenate([
        mut[genes].values,
        cna_amp.values,
        cna_del.values,
    ], axis=1).astype(np.float32)

    y = resp["response"].values.astype(np.float32)
    return X, y, list(common), list(genes)


def run_minimal_forward(X, y, verbose=False):
    """
    Build a tiny dense MLP (no biological mask) and run a forward pass.
    This is a structural proxy — not the full KEPPNet model.
    Real training requires Reactome + STRING data (see docs/).
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score

    logger.info(f"Input shape: {X.shape}  |  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, stratify=y.astype(int), random_state=42
    )

    in_dim = X.shape[1]

    class MiniNet(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 16), nn.Tanh(),
                nn.Linear(16, 8),  nn.Tanh(),
                nn.Linear(8, 1),   nn.Sigmoid(),
            )
        def forward(self, x):
            return self.net(x)

    model = MiniNet(in_dim)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.BCELoss()

    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr).unsqueeze(1)

    for epoch in range(50):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/50  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X_te)).numpy().ravel()

    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_te, probs)
    except Exception:
        auc = float("nan")
    f1 = f1_score(y_te.astype(int), preds, zero_division=0)

    return {"auc": round(float(auc), 4), "f1": round(float(f1), 4),
            "n_train": len(y_tr), "n_test": len(y_te)}


def show_attribution_output_format():
    logger.info("")
    logger.info("=" * 60)
    logger.info("Expected attribution output structure (full run):")
    logger.info("  results/attribution/<method>_<cancer>/extracted/")
    logger.info("    gradient_importance_0.csv  — gene-level scores")
    logger.info("    topk_layer_1.csv           — top-k nodes at layer 1")
    logger.info("    topk_summary.csv           — combined top-k summary")
    logger.info("    node_importance_graph_adjusted.csv")
    logger.info("    sankey.html                — interactive Sankey diagram")
    logger.info("")
    logger.info("Run attribution analysis (requires real data):")
    logger.info("  python src/interpretability/run_attribution.py \\")
    logger.info("      --method deeplift --cancer_type prostate")
    logger.info("=" * 60)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KEPPNet minimal demo using synthetic example data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cancer_type",
        choices=["prostate", "breast"],
        default="prostate",
        help="Cancer type label for display (default: prostate)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-epoch training loss",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KEPPNet Demo  |  knowledge-enhanced pathway-protein network")
    logger.info(f"Cancer type: {args.cancer_type}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Step 1: Loading example data from example_data/ ...")

    try:
        X, y, sample_ids, genes = load_example_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"  Loaded {len(sample_ids)} samples, {len(genes)} genes")
    logger.info(f"  Feature matrix: {X.shape}  (mut + cna_amp + cna_del)")
    logger.info("")
    logger.info("Step 2: Running minimal forward pass (dense proxy model) ...")
    logger.info("  NOTE: Full KEPPNet requires Reactome + STRING data.")
    logger.info("        See docs/data_preparation.md for setup.")
    logger.info("")

    metrics = run_minimal_forward(X, y, verbose=args.verbose)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Demo result (synthetic data — not representative of real performance):")
    logger.info(f"  AUC : {metrics['auc']}")
    logger.info(f"  F1  : {metrics['f1']}")
    logger.info(f"  Train samples: {metrics['n_train']}  |  Test samples: {metrics['n_test']}")
    logger.info("=" * 60)

    show_attribution_output_format()

    logger.info("")
    logger.info("Next steps with real data:")
    logger.info("  1. Prepare data:  python scripts/prepare_data/build_patient_summaries.py --cancer_type prostate")
    logger.info("  2. Train model:   python scripts/train.py --mode pathpronet --cancer_type prostate")
    logger.info("  3. Evaluate:      python scripts/evaluate.py --mode late_fusion --cancer_type prostate")
    logger.info("  4. Interpret:     python src/interpretability/run_attribution.py --method deeplift --cancer_type prostate")


if __name__ == "__main__":
    main()
