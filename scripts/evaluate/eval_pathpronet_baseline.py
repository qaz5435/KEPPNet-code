#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_pathpronet_strict_baseline.py

在 dual_strict_set 对应的样本口径上运行 PathProNet-only 基线。
最小改动原模型：通过 selected_samples 参数传入 patient list，
原模型的 get_train_validate_test() 会用 set.intersection 自动筛选。

用法:
    conda run -n bnet_env python scripts/run_pathpronet_strict_baseline.py \
        --cancer_type prostate --output_dir results/pathpronet_only

    conda run -n bnet_env python scripts/run_pathpronet_strict_baseline.py \
        --cancer_type breast --output_dir results/pathpronet_only
"""

import os
import sys
import csv
import logging
import argparse
import numpy as np

# 必须在任何 matplotlib/IPython import 之前设置无头后端
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

# 指定 GPU 7（与原项目 config.json 一致）

sys.path.insert(0, os.path.abspath("."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_dual_strict_ids(biolord_dir: str, cancer_type: str):
    """读取 dual_strict_set 中 in_dual_strict_set=1 的 patient_id，按 split 分组。"""
    subdir = "prad" if cancer_type == "prostate" else "brca"
    path = os.path.join(biolord_dir, subdir, "dual_strict_set.csv")
    splits = {"train": [], "val": [], "test": []}
    all_ids = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("in_dual_strict_set", "0")).strip() == "1":
                pid = row["patient_id"].strip()
                sp  = row["split"].strip()
                splits[sp].append(pid)
                all_ids.append(pid)
    logger.info(f"  dual_strict_set: train={len(splits['train'])}, "
                f"val={len(splits['val'])}, test={len(splits['test'])}")
    return splits, all_ids


def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, average_precision_score)
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc   = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "auprc": auprc}


def run_pathpronet(cancer_type: str, biolord_dir: str, output_dir: str):
    """
    运行 PathProNet 在 dual_strict_set 上。
    通过 Config + BnetData + BnetDataLoader + Model + Trainer 的标准流程，
    但在 get_train_validate_test 中通过 intersection 自动筛选 strict set 样本。
    """
    from src.config.configuration import Config
    from src.data.dataset import BnetData, set_cached_data_to_empty
    from src.data.dataload import BnetDataLoader
    from src.models.pathpronet import Model
    from src.training.trainer import Trainer, model_predict
    from src.utils.logger import init_logger
    from src.utils.utils import init_seed
    import torch

    # 选择最优模型配置
    if cancer_type == "prostate":
        model_name   = "prad-main"
        dataset_name = "prostate_kg"
    else:
        model_name   = "brca-main"
        dataset_name = "breast_kg"

    logger.info(f"  Model: {model_name}, Dataset: {dataset_name}")

    config = Config(model_name, dataset_name)
    config["save_res"] = False
    config["interpretability"] = False

    init_seed(config["random_seed"], config["reproducibility"])
    init_logger(config)

    set_cached_data_to_empty()
    dataset    = BnetData(config)
    dataloader = BnetDataLoader(config, dataset)
    model      = Model(config, dataset)
    trainer    = Trainer(config, dataloader, model)

    logger.info("  Training PathProNet ...")
    trainer.fix()

    # 获取 dual_strict_set 的 test 样本
    strict_splits, _ = load_dual_strict_ids(biolord_dir, cancer_type)
    strict_test_ids  = set(strict_splits["test"])
    strict_val_ids   = set(strict_splits["val"])

    # 从 dataloader 中筛选 strict set 对应的样本
    # dataloader.info_test_ 是 test 样本的 patient_id 列表
    test_info = dataloader.info_test_
    test_mask = np.array([pid in strict_test_ids for pid in test_info])
    val_info  = dataloader.info_validate_
    val_mask  = np.array([pid in strict_val_ids for pid in val_info])

    logger.info(f"  Test samples in strict set: {test_mask.sum()} / {len(test_mask)}")
    logger.info(f"  Val samples in strict set:  {val_mask.sum()} / {len(val_mask)}")

    device = config["device"]
    model.to(device)
    model.eval()

    # Test 预测
    X_test_strict = dataloader.x_test_[test_mask]
    y_test_strict = dataloader.y_test_[test_mask]
    X_test_t = torch.tensor(X_test_strict, dtype=torch.float32)
    y_prob_test = model_predict(model, X_test_t, device).cpu().numpy().ravel()
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    y_true_test = y_test_strict.ravel().astype(int)

    # Val 预测（供 late fusion 使用）
    X_val_strict = dataloader.x_validate_[val_mask]
    y_val_strict = dataloader.y_validate_[val_mask]
    X_val_t = torch.tensor(X_val_strict, dtype=torch.float32)
    y_prob_val = model_predict(model, X_val_t, device).cpu().numpy().ravel()
    y_pred_val = (y_prob_val >= 0.5).astype(int)
    y_true_val = y_val_strict.ravel().astype(int)

    metrics = compute_metrics(y_true_test, y_pred_test, y_prob_test)
    logger.info(f"  PathProNet strict test: acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} auprc={metrics['auprc']:.4f}")

    # 保存 metrics
    ct_short = "prad" if cancer_type == "prostate" else "brca"
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{ct_short}_metrics.csv")
    row = {"model": "PathProNet", "set": "dual_strict_set", "cancer_type": cancer_type,
           "train_n": len(dataloader.y_train), "val_n": int(val_mask.sum()),
           "test_n": int(test_mask.sum())}
    row.update({k: round(v, 4) for k, v in metrics.items()})
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"  Metrics saved: {metrics_path}")

    # 保存预测概率（供 late fusion 使用）
    prob_dir = os.path.join(output_dir, "probs", ct_short)
    os.makedirs(prob_dir, exist_ok=True)

    val_ids_strict = [pid for pid, m in zip(val_info, val_mask) if m]
    test_ids_strict = [pid for pid, m in zip(test_info, test_mask) if m]

    val_rows = [{"patient_id": pid, "true_label": int(y), "pathpronet_prob": float(p)}
                for pid, y, p in zip(val_ids_strict, y_true_val, y_prob_val)]
    with open(os.path.join(prob_dir, "val_probs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "true_label", "pathpronet_prob"])
        writer.writeheader(); writer.writerows(val_rows)

    test_rows = [{"patient_id": pid, "true_label": int(y), "pathpronet_prob": float(p)}
                 for pid, y, p in zip(test_ids_strict, y_true_test, y_prob_test)]
    with open(os.path.join(prob_dir, "test_probs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "true_label", "pathpronet_prob"])
        writer.writeheader(); writer.writerows(test_rows)

    logger.info(f"  Probs saved: {prob_dir}")
    set_cached_data_to_empty()

    return metrics, val_ids_strict, y_prob_val, y_true_val, test_ids_strict, y_prob_test, y_true_test


def main():
    parser = argparse.ArgumentParser(description="Run PathProNet on dual_strict_set")
    parser.add_argument("--cancer_type",  required=True, choices=["prostate", "breast"])
    parser.add_argument("--biolord_dir",  default="data/biolord_inputs")
    parser.add_argument("--output_dir",   default="results/pathpronet_only")
    args = parser.parse_args()

    logger.info(f"=== PathProNet strict baseline: {args.cancer_type.upper()} ===")
    run_pathpronet(args.cancer_type, args.biolord_dir, args.output_dir)
    logger.info(f"=== DONE ===")


if __name__ == "__main__":
    main()
