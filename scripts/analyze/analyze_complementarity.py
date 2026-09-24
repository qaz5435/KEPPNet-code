#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analyze_branch_complementarity.py

统计两个分支的互补性：
  - 两个分支都预测正确
  - PathProNet 对但 BioLORD 错
  - BioLORD 对但 PathProNet 错
  - 两个分支都错

用法:
    conda run -n biolord_gpu310 python scripts/analyze_branch_complementarity.py \
        --output_dir results/complementarity
"""

import os
import csv
import logging
import argparse
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_probs(path: str, prob_col: str, threshold: float = 0.5):
    """返回 {patient_id: (true_label, pred_label, prob)}。"""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid  = row["patient_id"].strip()
            true = int(row["true_label"])
            prob = float(row[prob_col])
            pred = int(prob >= threshold)
            data[pid] = (true, pred)
    return data


def analyze(cancer_type: str, pathpronet_dir: str, llm_dir: str, output_dir: str, lines: list):
    ct_short = "prad" if cancer_type == "prostate" else "brca"

    def log(msg=""):
        logger.info(msg)
        lines.append(msg)

    log(f"\n{'='*60}")
    log(f"  {cancer_type.upper()} Complementarity Analysis")
    log(f"{'='*60}")

    # 加载 PathProNet 测试集预测
    bio_path = os.path.join(pathpronet_dir, "probs", ct_short, "test_probs.csv")
    llm_path = os.path.join(llm_dir, "probs", ct_short, "MLP_test_probs.csv")

    if not os.path.exists(bio_path):
        log(f"  [ERROR] PathProNet probs not found: {bio_path}")
        return
    if not os.path.exists(llm_path):
        log(f"  [ERROR] LLM probs not found: {llm_path}")
        return

    bio_data = load_probs(bio_path, "pathpronet_prob")
    llm_data = load_probs(llm_path, "llm_prob")

    common = sorted(set(bio_data.keys()) & set(llm_data.keys()))
    log(f"  Common test patients: {len(common)}")

    # 四类统计
    both_correct   = []
    bio_only       = []
    llm_only       = []
    both_wrong     = []

    rows = []
    for pid in common:
        true_label, bio_pred = bio_data[pid]
        _,          llm_pred = llm_data[pid]
        bio_correct = int(bio_pred == true_label)
        llm_correct = int(llm_pred == true_label)

        if bio_correct and llm_correct:
            category = "both_correct"
            both_correct.append(pid)
        elif bio_correct and not llm_correct:
            category = "pathpronet_only_correct"
            bio_only.append(pid)
        elif not bio_correct and llm_correct:
            category = "llm_only_correct"
            llm_only.append(pid)
        else:
            category = "both_wrong"
            both_wrong.append(pid)

        rows.append({
            "patient_id":    pid,
            "true_label":    true_label,
            "pathpronet_pred": bio_pred,
            "llm_pred":      llm_pred,
            "pathpronet_correct": bio_correct,
            "llm_correct":   llm_correct,
            "category":      category,
        })

    log(f"\n  Both correct              : {len(both_correct)} ({len(both_correct)/len(common)*100:.1f}%)")
    log(f"  PathProNet only correct   : {len(bio_only)} ({len(bio_only)/len(common)*100:.1f}%)")
    log(f"  BioLORD only correct      : {len(llm_only)} ({len(llm_only)/len(common)*100:.1f}%)")
    log(f"  Both wrong                : {len(both_wrong)} ({len(both_wrong)/len(common)*100:.1f}%)")

    complementary = len(bio_only) + len(llm_only)
    log(f"\n  Complementary samples     : {complementary} ({complementary/len(common)*100:.1f}%)")
    log(f"  => {'Strong' if complementary/len(common) > 0.15 else 'Moderate'} complementarity signal")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{ct_short}_complementarity.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log(f"\n  Saved: {out_path}")

    return {
        "both_correct": len(both_correct),
        "pathpronet_only": len(bio_only),
        "llm_only": len(llm_only),
        "both_wrong": len(both_wrong),
        "total": len(common),
    }


def main():
    parser = argparse.ArgumentParser(description="Branch complementarity analysis")
    parser.add_argument("--pathpronet_dir", default="results/pathpronet_only")
    parser.add_argument("--llm_dir",        default="results/llm_only")
    parser.add_argument("--output_dir",     default="results/complementarity")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lines = []

    def log(msg=""):
        logger.info(msg)
        lines.append(msg)

    log("=" * 60)
    log("  BRANCH COMPLEMENTARITY ANALYSIS")
    log("=" * 60)

    results = {}
    for ct in ["prostate", "breast"]:
        r = analyze(ct, args.pathpronet_dir, args.llm_dir, args.output_dir, lines)
        if r:
            results[ct] = r

    # 总结
    log(f"\n{'='*60}")
    log("  SUMMARY")
    log(f"{'='*60}")
    for ct, r in results.items():
        log(f"  {ct.upper()} (n={r['total']}):")
        log(f"    Both correct    : {r['both_correct']}")
        log(f"    PathProNet only : {r['pathpronet_only']}")
        log(f"    BioLORD only    : {r['llm_only']}")
        log(f"    Both wrong      : {r['both_wrong']}")

    # 保存报告
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
