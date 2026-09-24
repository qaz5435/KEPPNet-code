#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analyze_cross_branch_support.py

跨分支有效性支持分析：
1. 预测一致性分析（含高置信度统计）
2. feature fusion vs late fusion 错误修正分析
3. 生物学主题支持分析（pathway overlap）

用法:
    conda run -n bnet_env python scripts/analyze_cross_branch_support.py
"""
import os, csv, logging, sys
import numpy as np
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

OUTPUT_DIR = "results/cross_branch_support"


def load_probs(path, prob_col, threshold=0.5):
    data = {}
    if not os.path.exists(path): return data
    with open(path) as f:
        for row in csv.DictReader(f):
            pid  = row["patient_id"].strip()
            true = int(row["true_label"])
            prob = float(row[prob_col])
            pred = int(prob >= threshold)
            data[pid] = {"true": true, "pred": pred, "prob": prob}
    return data


def consistency_analysis(bio_data, llm_data, cancer_type, lines):
    def log(msg=""): logger.info(msg); lines.append(msg)

    common = sorted(set(bio_data) & set(llm_data))
    log(f"\n[{cancer_type.upper()}] Prediction Consistency Analysis (n={len(common)})")

    both_correct = both_wrong = bio_only = llm_only = 0
    high_conf_agree = high_conf_conflict = 0
    CONF_THRESH = 0.7

    for pid in common:
        b = bio_data[pid]; l = llm_data[pid]
        bc = int(b["pred"] == b["true"]); lc = int(l["pred"] == l["true"])
        if bc and lc:     both_correct += 1
        elif bc and not lc: bio_only += 1
        elif not bc and lc: llm_only += 1
        else:               both_wrong += 1

        # 高置信度
        b_conf = abs(b["prob"] - 0.5) * 2
        l_conf = abs(l["prob"] - 0.5) * 2
        if b_conf > CONF_THRESH and l_conf > CONF_THRESH:
            if b["pred"] == l["pred"]: high_conf_agree += 1
            else:                      high_conf_conflict += 1

    n = len(common)
    log(f"  Both correct:              {both_correct} ({both_correct/n*100:.1f}%)")
    log(f"  PathProNet only correct:   {bio_only} ({bio_only/n*100:.1f}%)")
    log(f"  BioLORD only correct:      {llm_only} ({llm_only/n*100:.1f}%)")
    log(f"  Both wrong:                {both_wrong} ({both_wrong/n*100:.1f}%)")
    log(f"  Complementary rate:        {(bio_only+llm_only)/n*100:.1f}%")
    log(f"  High-conf agree:           {high_conf_agree}")
    log(f"  High-conf conflict:        {high_conf_conflict}")
    return {"both_correct": both_correct, "bio_only": bio_only,
            "llm_only": llm_only, "both_wrong": both_wrong, "n": n}


def fusion_correction_analysis(bio_data, llm_data, late_data, feat_data, cancer_type, lines):
    def log(msg=""): logger.info(msg); lines.append(msg)

    common = sorted(set(bio_data) & set(llm_data) & set(late_data) & set(feat_data))
    log(f"\n[{cancer_type.upper()}] Fusion Correction Analysis (n={len(common)})")

    late_fixed = feat_fixed = late_broke = feat_broke = 0
    for pid in common:
        b = bio_data[pid]; l = llm_data[pid]
        la = late_data[pid]; fe = feat_data[pid]
        base_wrong = int(b["pred"] != b["true"]) or int(l["pred"] != l["true"])
        late_correct = int(la["pred"] == la["true"])
        feat_correct = int(fe["pred"] == fe["true"])
        bio_correct  = int(b["pred"] == b["true"])

        if not bio_correct and late_correct: late_fixed += 1
        if not bio_correct and feat_correct: feat_fixed += 1
        if bio_correct and not late_correct: late_broke += 1
        if bio_correct and not feat_correct: feat_broke += 1

    log(f"  Late fusion fixed (vs PathProNet):    {late_fixed}")
    log(f"  Feature fusion fixed (vs PathProNet): {feat_fixed}")
    log(f"  Late fusion broke (vs PathProNet):    {late_broke}")
    log(f"  Feature fusion broke (vs PathProNet): {feat_broke}")
    log(f"  Net gain late:    {late_fixed - late_broke}")
    log(f"  Net gain feature: {feat_fixed - feat_broke}")


def pathway_support_analysis(cancer_type, lines):
    """
    轻量版生物学主题支持分析：
    比较 PathProNet 高权重通路 vs BioLORD 文本中的 top pathways 的重叠率。
    """
    def log(msg=""): logger.info(msg); lines.append(msg)

    subdir = "prad" if cancer_type == "prostate" else "brca"
    pw_file = f"data/biolord_inputs/{subdir}/patient_top_pathways.csv"
    topk_dir = f"results/attribution_stability"

    log(f"\n[{cancer_type.upper()}] Biological Theme Support Analysis")

    # 读取患者 top pathways（来自 BioLORD 文本分支）
    patient_pws = {}
    if os.path.exists(pw_file):
        with open(pw_file) as f:
            for row in csv.DictReader(f):
                pids = row["patient_id"].strip()
                pws = [p.strip() for p in row.get("top_pathways","").split(";") if p.strip()]
                patient_pws[pids] = set(pws)
        log(f"  Loaded top pathways for {len(patient_pws)} patients")

    # 读取 PathProNet 归因分析的 top nodes（如果有）
    topk_nodes = set()
    for root, dirs, files in os.walk(topk_dir):
        for fname in files:
            if fname.startswith("topk_layer") and fname.endswith(".csv"):
                try:
                    with open(os.path.join(root, fname)) as f:
                        for row in csv.DictReader(f):
                            node = row.get("node","").strip()
                            if node.startswith("R-HSA-"):
                                topk_nodes.add(node)
                except: pass

    if topk_nodes:
        log(f"  PathProNet top pathway nodes (from attribution): {len(topk_nodes)}")
        # 计算每个患者的 BioLORD top pathways 与 PathProNet top nodes 的重叠
        overlaps = []
        for pid, pws in patient_pws.items():
            overlap = len(pws & topk_nodes)
            overlaps.append(overlap)
        if overlaps:
            log(f"  Mean pathway overlap per patient: {np.mean(overlaps):.2f}")
            log(f"  Patients with ≥1 overlap: {sum(1 for o in overlaps if o>0)} ({sum(1 for o in overlaps if o>0)/len(overlaps)*100:.1f}%)")
            log(f"  Patients with ≥3 overlap: {sum(1 for o in overlaps if o>=3)} ({sum(1 for o in overlaps if o>=3)/len(overlaps)*100:.1f}%)")
    else:
        log(f"  No PathProNet attribution results found in {topk_dir}")
        log(f"  Falling back to cohort-level top pathway frequency analysis ...")

        # 队列级：统计 BioLORD 文本中最高频的通路
        pw_counter = Counter()
        for pws in patient_pws.values():
            pw_counter.update(pws)
        top10 = pw_counter.most_common(10)
        log(f"  Top 10 most frequent pathways in {cancer_type} cohort (BioLORD text branch):")
        for pw, cnt in top10:
            log(f"    {pw}: {cnt} patients ({cnt/len(patient_pws)*100:.1f}%)")

    # 读取 PathProNet 的 topk_nodes_summary（如果有）
    summary_files = [
        "ProPath_NET/result/topk_nodes_summary.csv",
        "ProPath_NET/result/topk_nodes_summary_breast.csv",
    ]
    for sf in summary_files:
        if os.path.exists(sf):
            try:
                with open(sf) as f:
                    rows = list(csv.DictReader(f))
                pathpronet_top = set()
                for row in rows[:20]:
                    node = row.get("node","").strip()
                    if node.startswith("R-HSA-"):
                        pathpronet_top.add(node)
                if pathpronet_top and patient_pws:
                    overlaps = [len(pws & pathpronet_top) for pws in patient_pws.values()]
                    log(f"\n  Cross-branch overlap with PathProNet top nodes ({sf}):")
                    log(f"    PathProNet top nodes: {len(pathpronet_top)}")
                    log(f"    Mean overlap per patient: {np.mean(overlaps):.2f}")
                    log(f"    Patients with ≥1 overlap: {sum(1 for o in overlaps if o>0)/len(overlaps)*100:.1f}%")
            except: pass


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []
    def log(msg=""): logger.info(msg); lines.append(msg)

    log("="*60)
    log("  CROSS-BRANCH VALIDITY SUPPORT ANALYSIS")
    log("="*60)

    for cancer_type in ["prostate", "breast"]:
        ct_short = "prad" if cancer_type == "prostate" else "brca"

        bio_path  = f"results/pathpronet_only/probs/{ct_short}/test_probs.csv"
        llm_path  = f"results/llm_only/probs/{ct_short}/MLP_test_probs.csv"
        late_path = f"results/fusion_late/probs/{ct_short}/test_probs.csv"
        feat_path = f"results/feature_fusion/probs/{ct_short}/test_probs.csv"

        bio_data  = load_probs(bio_path,  "pathpronet_prob")
        llm_data  = load_probs(llm_path,  "llm_prob")
        late_data = load_probs(late_path, "fusion_prob")
        feat_data = load_probs(feat_path, "fusion_prob")

        consistency_analysis(bio_data, llm_data, cancer_type, lines)
        if late_data and feat_data:
            fusion_correction_analysis(bio_data, llm_data, late_data, feat_data, cancer_type, lines)
        pathway_support_analysis(cancer_type, lines)

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "cross_branch_support_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
