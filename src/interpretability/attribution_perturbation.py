#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/interpretability/attribution_perturbation.py

Attribution-guided perturbation analysis for KEPPNet.

Procedure:
1. Run attribution (DeepLIFT / IG / GradientSHAP) to rank nodes by importance.
2. Progressively mask top-k nodes (set their activations to zero).
3. Re-evaluate model performance after each masking step.
4. Plot performance degradation curve to validate attribution quality.

Usage:
    python src/interpretability/attribution_perturbation.py \
        --cancer_type prostate \
        --method deeplift \
        --layer h2 \
        --topk_steps 1,3,5,10,20 \
        --output_dir results/perturbation
"""

import os
import sys
import csv
import argparse
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def load_topk_summary(attribution_dir: str, layer: str) -> pd.DataFrame:
    """Load top-k node summary from attribution output."""
    # Try layer-specific file first
    layer_num = int(layer[1:]) + 1 if layer.startswith("h") and layer[1:].isdigit() else None
    if layer_num is not None:
        path = os.path.join(attribution_dir, "extracted", f"topk_layer_{layer_num}.csv")
        if os.path.exists(path):
            return pd.read_csv(path, index_col=0)
    # Fall back to summary
    summary_path = os.path.join(attribution_dir, "extracted", "topk_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        if layer_num is not None:
            df = df[df["layer"] == layer_num]
        return df
    raise FileNotFoundError(f"No attribution output found in {attribution_dir}/extracted/")


def mask_nodes_and_evaluate(
    model,
    dataloader,
    config,
    layer: str,
    nodes_to_mask: list,
) -> dict:
    """
    Mask specified nodes in the given layer by zeroing their output,
    then evaluate model performance on the test set.

    Uses a forward hook to intercept and zero the activations.
    """
    import torch
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

    layer_idx = int(layer[1:]) if layer.startswith("h") and layer[1:].isdigit() else None
    if layer_idx is None:
        raise ValueError(f"Unsupported layer format: {layer}. Expected h0, h1, h2, ...")

    # Build node -> column index mapping
    feature_names = model.feature_names.get(layer, [])
    node_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    mask_indices = [node_to_idx[n] for n in nodes_to_mask if n in node_to_idx]

    if not mask_indices:
        logger.warning(f"None of the specified nodes found in layer {layer}")

    # Register hook
    hooks = []
    def make_hook(indices):
        def hook_fn(module, input, output):
            out = output.clone()
            out[:, indices] = 0.0
            return out
        return hook_fn

    if layer_idx < len(model.hidden_layers):
        target_layer = model.hidden_layers[layer_idx]
    else:
        target_layer = model.h0

    if mask_indices:
        h = target_layer.register_forward_hook(make_hook(mask_indices))
        hooks.append(h)

    model.eval()
    all_probs, all_labels = [], []
    X = torch.tensor(dataloader.x_test_, dtype=torch.float32)
    y = dataloader.y_test_

    with torch.no_grad():
        out = model(X)
        probs = out[:, -1].cpu().numpy()

    for h in hooks:
        h.remove()

    all_probs = probs.flatten()
    all_labels = y.flatten().astype(int)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    try:
        auprc = average_precision_score(all_labels, all_probs)
    except Exception:
        auprc = float("nan")

    th = 0.5
    preds = (all_probs >= th).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)

    return {"auc": auc, "auprc": auprc, "f1": f1, "n_masked": len(mask_indices)}


def run_perturbation(
    cancer_type: str,
    method: str,
    layer: str,
    topk_steps: list,
    attribution_dir: str,
    output_dir: str,
):
    """Full perturbation analysis pipeline."""
    from src.data.dataload import BnetDataLoader
    from src.data.dataset import BnetData, set_cached_data_to_empty
    from src.config.configuration import Config
    from src.models.pathpronet import Model
    from src.utils.logger import init_logger
    from src.utils.utils import init_seed
    from src.training.trainer import Trainer

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

    cfg = BEST_CONFIGS[cancer_type]
    set_cached_data_to_empty()
    config = Config(cfg["model_name"], cfg["dataset_name"])
    config["study_names"] = cfg["study_names"]
    config["output_dir"] = output_dir

    init_seed(config["random_seed"], config["reproducibility"])
    init_logger(config)
    dataset    = BnetData(config)
    dataloader = BnetDataLoader(config, dataset)
    model      = Model(config, dataset)
    trainer    = Trainer(config, dataloader, model)
    trainer.fix()

    # Load ranked nodes from attribution output
    topk_df = load_topk_summary(attribution_dir, layer)
    if "node" in topk_df.columns:
        ranked_nodes = list(topk_df.sort_values("rank")["node"].astype(str))
    else:
        ranked_nodes = list(topk_df.index.astype(str))

    logger.info(f"Loaded {len(ranked_nodes)} ranked nodes from {attribution_dir}")

    # Baseline (no masking)
    baseline_metrics = mask_nodes_and_evaluate(model, dataloader, config, layer, [])
    logger.info(f"Baseline: AUC={baseline_metrics['auc']:.4f}, F1={baseline_metrics['f1']:.4f}")

    results = [{"topk": 0, "method": "baseline", **baseline_metrics}]

    for k in sorted(topk_steps):
        nodes = ranked_nodes[:k]
        metrics = mask_nodes_and_evaluate(model, dataloader, config, layer, nodes)
        logger.info(f"Mask top-{k}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
        results.append({"topk": k, "method": method, **metrics})

    # Also run random masking as control
    import random
    all_nodes = list(model.feature_names.get(layer, []))
    for k in sorted(topk_steps):
        random_nodes = random.sample(all_nodes, min(k, len(all_nodes)))
        metrics = mask_nodes_and_evaluate(model, dataloader, config, layer, random_nodes)
        results.append({"topk": k, "method": "random", **metrics})

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"perturbation_{cancer_type}_{layer}_{method}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    logger.info(f"Perturbation results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="KEPPNet attribution-guided perturbation analysis")
    parser.add_argument("--cancer_type", choices=["prostate", "breast"], required=True)
    parser.add_argument("--method", choices=["deeplift", "integratedgradients", "gradientshap"],
                        default="deeplift")
    parser.add_argument("--layer", default="h2",
                        help="Layer to perturb (e.g. h0, h1, h2)")
    parser.add_argument("--topk_steps", default="1,3,5,10,20",
                        help="Comma-separated list of top-k values to mask")
    parser.add_argument("--attribution_dir", default=None,
                        help="Directory containing attribution outputs (default: results/attribution/<method>_<cancer>)")
    parser.add_argument("--output_dir", default="results/perturbation")
    args = parser.parse_args()

    topk_steps = [int(x) for x in args.topk_steps.split(",")]
    if args.attribution_dir is None:
        args.attribution_dir = f"results/attribution/{args.method}_{args.cancer_type}"

    run_perturbation(
        cancer_type=args.cancer_type,
        method=args.method,
        layer=args.layer,
        topk_steps=topk_steps,
        attribution_dir=args.attribution_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
