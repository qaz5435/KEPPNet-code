#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_late_fusion_baseline.py

Late fusion: p_final = λ * p_bio + (1-λ) * p_llm
在验证集上搜索最优 λ，在测试集上汇报结果。
基于 dual_strict_set，使用 PathProNet 和 BioLORD-MLP 的预测概率。

用法:
    conda run -n biolord_gpu310 python scripts/run_late_fusion_baseline.py \
        --cancer_type prostate --output_dir results/fusion_late

    conda run -n biolord_gpu310 python scripts/run_late_fusion_baseline.py \
        --cancer_type breast --output_dir results/fusion_late
"""

import os
import sys
import csv
import logging
import argparse
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_probs(path: str, prob_col: str):
    """读取概率文件，返回 {patient_id: (true_label, prob)}。"""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"].strip()
            data[pid] = (int(row["true_label"]), float(row[prob_col]))
    return data


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


def fuse_and_eval(bio_data, llm_data, lam):
    """在共同患者上做 late fusion，返回 metrics。"""
    common = set(bio_data.keys()) & set(llm_data.keys())
    if not common:
        return None, [], [], []
    pids = sorted(common)
    mismatched = [p for p in pids if bio_data[p][0] != llm_data[p][0]]
    if mismatched:
        raise ValueError(
            f"Label mismatch between structured and semantic predictions for "
            f"{len(mismatched)} patients"
        )
    y_true = np.array([bio_data[p][0] for p in pids])
    p_bio  = np.array([bio_data[p][1] for p in pids])
    p_llm  = np.array([llm_data[p][1] for p in pids])
    p_fuse = lam * p_bio + (1 - lam) * p_llm
    y_pred = (p_fuse >= 0.5).astype(int)
    return compute_metrics(y_true, y_pred, p_fuse), pids, y_true, p_fuse


def main():
    parser = argparse.ArgumentParser(description="Late fusion baseline")
    parser.add_argument("--cancer_type",    required=True, choices=["prostate", "breast"])
    parser.add_argument("--llm_model",      default="MLP",
                        help="Which LLM model to use for fusion (LR/LinearSVM/MLP)")
    parser.add_argument("--pathpronet_dir", default="results/pathpronet_only")
    parser.add_argument("--llm_dir",        default="results/llm_only")
    parser.add_argument("--output_dir",     default="results/fusion_late")
    args = parser.parse_args()

    ct       = args.cancer_type
    ct_short = "prad" if ct == "prostate" else "brca"
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"=== Late Fusion: {ct.upper()} (LLM model: {args.llm_model}) ===")

    # 加载 PathProNet 概率
    bio_val_path  = os.path.join(args.pathpronet_dir, "probs", ct_short, "val_probs.csv")
    bio_test_path = os.path.join(args.pathpronet_dir, "probs", ct_short, "test_probs.csv")
    bio_val  = load_probs(bio_val_path,  "pathpronet_prob")
    bio_test = load_probs(bio_test_path, "pathpronet_prob")
    logger.info(f"  PathProNet val={len(bio_val)}, test={len(bio_test)}")

    # 加载 LLM 概率（使用 dual_strict_set 上的结果）
    llm_val_path  = os.path.join(args.llm_dir, "probs", ct_short, f"{args.llm_model}_val_probs.csv")
    llm_test_path = os.path.join(args.llm_dir, "probs", ct_short, f"{args.llm_model}_test_probs.csv")
    llm_val  = load_probs(llm_val_path,  "llm_prob")
    llm_test = load_probs(llm_test_path, "llm_prob")
    logger.info(f"  LLM ({args.llm_model}) val={len(llm_val)}, test={len(llm_test)}")

    # 在验证集上搜索最优 λ（0.0 到 1.0，步长 0.1）
    lambdas = [round(x * 0.1, 1) for x in range(11)]
    best_lam = 0.5
    best_val_auc = -1
    val_results = []

    logger.info("  Searching λ on validation set ...")
    for lam in lambdas:
        m, _, _, _ = fuse_and_eval(bio_val, llm_val, lam)
        if m is None:
            continue
        val_results.append({"lambda": lam, **{k: round(v, 4) for k, v in m.items()}})
        logger.info(f"    λ={lam:.1f}: val_auc={m['auc']:.4f} val_f1={m['f1']:.4f}")
        if m["auc"] > best_val_auc:
            best_val_auc = m["auc"]
            best_lam = lam

    logger.info(f"  Best λ = {best_lam} (val_auc={best_val_auc:.4f})")

    # 在测试集上用最优 λ 评估
    test_m, test_pids, test_true, test_prob = fuse_and_eval(bio_test, llm_test, best_lam)
    if test_m is None:
        logger.error("  No common patients in test set!")
        return

    logger.info(f"  Test (λ={best_lam}): acc={test_m['accuracy']:.4f} "
                f"f1={test_m['f1']:.4f} auc={test_m['auc']:.4f} auprc={test_m['auprc']:.4f}")

    # 保存结果
    metrics_path = os.path.join(args.output_dir, f"{ct_short}_metrics.csv")
    row = {"model": f"LateFusion_λ={best_lam}", "llm_model": args.llm_model,
           "set": "dual_strict_set", "cancer_type": ct,
           "best_lambda": best_lam, "val_auc": round(best_val_auc, 4),
           "test_n": len(test_pids)}
    row.update({k: round(v, 4) for k, v in test_m.items()})
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"  Metrics saved: {metrics_path}")

    # 保存 val λ 搜索结果
    val_path = os.path.join(args.output_dir, f"{ct_short}_lambda_search.csv")
    with open(val_path, "w", newline="", encoding="utf-8") as f:
        if val_results:
            writer = csv.DictWriter(f, fieldnames=list(val_results[0].keys()))
            writer.writeheader()
            writer.writerows(val_results)
    logger.info(f"  Lambda search saved: {val_path}")

    # 保存测试集预测概率（供互补性分析使用）
    prob_dir = os.path.join(args.output_dir, "probs", ct_short)
    os.makedirs(prob_dir, exist_ok=True)
    test_rows = [{"patient_id": pid, "true_label": int(y), "fusion_prob": float(p),
                  "fusion_pred": int(p >= 0.5)}
                 for pid, y, p in zip(test_pids, test_true, test_prob)]
    with open(os.path.join(prob_dir, "test_probs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "true_label", "fusion_prob", "fusion_pred"])
        writer.writeheader(); writer.writerows(test_rows)

    logger.info(f"=== DONE: {ct.upper()} ===")


if __name__ == "__main__":
    main()
