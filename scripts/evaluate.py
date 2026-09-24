#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/evaluate.py — KEPPNet unified evaluation entry point.

Usage:
    # PathProNet structural branch only
    python scripts/evaluate.py --mode pathpronet --cancer_type prostate

    # BioLORD semantic branch only (LR / LinearSVM / MLP)
    python scripts/evaluate.py --mode llm_only --cancer_type prostate

    # Late fusion from previously exported branch probabilities
    python scripts/evaluate.py --mode late_fusion --cancer_type prostate

    # Feature-level fusion: MLP(concat(h_bio, h_llm))
    python scripts/evaluate.py --mode feature_fusion --cancer_type breast

All modes require pre-computed data in data/biolord_inputs/.
See docs/data_preparation.md for setup instructions.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(
        description="KEPPNet — evaluate structural branch, semantic branch, or fusion models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["pathpronet", "llm_only", "late_fusion", "feature_fusion"],
        required=True,
        help=(
            "pathpronet: PathProNet structural branch baseline; "
            "llm_only: BioLORD-only baselines (LR/SVM/MLP); "
            "late_fusion: weighted fusion of exported probabilities; "
            "feature_fusion: concatenation + MLP fusion (ablation)"
        ),
    )
    parser.add_argument(
        "--cancer_type",
        choices=["prostate", "breast"],
        required=True,
        help="Cancer type to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: results/<mode>)",
    )
    parser.add_argument(
        "--biolord_dir",
        default="data/biolord_inputs",
        help="Directory containing BioLORD embeddings (default: data/biolord_inputs)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("results", args.mode)
    os.makedirs(args.output_dir, exist_ok=True)

    # Delegate to sub-scripts, passing args via sys.argv
    base_argv = [sys.argv[0],
                 "--cancer_type", args.cancer_type,
                 "--output_dir", args.output_dir]

    if args.mode == "pathpronet":
        from scripts.evaluate.eval_pathpronet_baseline import main as run
        sys.argv = base_argv + ["--biolord_dir", args.biolord_dir]
        run()

    elif args.mode == "llm_only":
        from scripts.evaluate.eval_llm_only_baselines import main as run
        sys.argv = base_argv + ["--biolord_dir", args.biolord_dir]
        run()

    elif args.mode == "late_fusion":
        from scripts.evaluate.eval_late_fusion import main as run
        sys.argv = base_argv
        run()

    elif args.mode == "feature_fusion":
        from scripts.evaluate.eval_feature_fusion import main as run
        sys.argv = [sys.argv[0], "--cancer_type", args.cancer_type,
                    "--output_dir", args.output_dir]
        run()


if __name__ == "__main__":
    main()
