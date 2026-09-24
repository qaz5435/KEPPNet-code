#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/interpretability/run_attribution.py

Attribution analysis entry point for KEPPNet structural branch (PathProNet).
Supports three methods: DeepLIFT, Integrated Gradients, GradientSHAP.

Usage:
    python src/interpretability/run_attribution.py \
        --method deeplift \
        --cancer_type prostate \
        --output_dir results/attribution

    python src/interpretability/run_attribution.py \
        --method integratedgradients \
        --cancer_type breast \
        --baseline mean

    python src/interpretability/run_attribution.py \
        --method gradientshap \
        --cancer_type prostate \
        --topk 15
"""

import os
import sys
import argparse
import logging

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.abspath("."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BEST_CONFIGS = {
    "prostate": {
        "model_name":   "prad-main",
        "dataset_name": "prostate_kg",
        "study_names":  ["prad_p1000"],
    },
    "breast": {
        "model_name":   "brca-main",
        "dataset_name": "breast_kg",
        "study_names":  ["brca_igr_2015", "brca_mbcproject_wagle_2017",
                         "brca_mbcproject_2022", "brca_tcga_pan_can_atlas_2018"],
    },
}


def run_attribution(cancer_type: str, method: str, baseline: str,
                    topk: int, output_dir: str):
    from src.data.dataload import BnetDataLoader
    from src.data.dataset import BnetData, set_cached_data_to_empty
    from src.config.configuration import Config
    from src.models.pathpronet import Model
    from src.utils.logger import init_logger
    from src.utils.utils import init_seed
    from src.training.trainer import Trainer
    from src.interpretability.attribution_methods import interpret_propathnet_model

    cfg = BEST_CONFIGS[cancer_type]
    set_cached_data_to_empty()
    config = Config(cfg["model_name"], cfg["dataset_name"])
    config["study_names"] = cfg["study_names"]
    config["output_dir"] = output_dir
    # attribution method overrides
    config["method_name"] = method
    config["feature_important_name"] = method
    config["baseline"] = baseline
    config["topk_per_layer"] = topk
    config["generate_sankey"] = True

    init_seed(config["random_seed"], config["reproducibility"])
    init_logger(config)

    dataset = BnetData(config)
    dataloader = BnetDataLoader(config, dataset)
    model = Model(config, dataset)
    trainer = Trainer(config, dataloader, model)
    trainer.fix()

    logger.info(f"Running {method} attribution for {cancer_type} -> {output_dir}")
    result = interpret_propathnet_model(config, model, dataloader)
    logger.info(f"Attribution complete. Outputs in {output_dir}/extracted/")
    return result


def main():
    parser = argparse.ArgumentParser(description="KEPPNet attribution analysis")
    parser.add_argument("--method", choices=["deeplift", "integratedgradients", "gradientshap"],
                        default="deeplift", help="Attribution method")
    parser.add_argument("--cancer_type", choices=["prostate", "breast"],
                        required=True, help="Cancer type")
    parser.add_argument("--baseline", choices=["zero", "mean"], default="zero",
                        help="Baseline for attribution (zero or mean; mean approximates GradientSHAP)")
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-k nodes to extract per layer")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/attribution/<method>_<cancer>)")
    args = parser.parse_args()

    # gradientshap uses mean baseline by default
    if args.method == "gradientshap" and args.baseline == "zero":
        args.baseline = "mean"
        logger.info("gradientshap: switching baseline to 'mean'")

    if args.output_dir is None:
        args.output_dir = f"results/attribution/{args.method}_{args.cancer_type}"

    os.makedirs(args.output_dir, exist_ok=True)
    run_attribution(
        cancer_type=args.cancer_type,
        method=args.method,
        baseline=args.baseline,
        topk=args.topk,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
