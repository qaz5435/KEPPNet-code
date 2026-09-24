#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/define_experiment_sets.py

重新定义三类严格样本集合，分别针对 PRAD 和 BRCA：
  1. llm_text_set.csv   - BioLORD-only 可用
  2. structured_set.csv - PathProNet-only 可用（基于原模型真实逻辑）
  3. dual_strict_set.csv - 双分支严格共同集合

原模型真实处理逻辑（来自 src/data/dataset.py）：
  - data_type = ["mut_important", "cnv_del", "cnv_amp"]
  - combine_type = "union"（基因列取并集）
  - combine() 内部：pd.concat(..., join='inner', axis=1)
    => 样本行取交集：必须同时出现在 mut_important 矩阵 AND CNA 矩阵中
  - load_data() 内部：data.join(labels, how='inner')
    => 样本必须在 data 矩阵中有行
  - 结论：structured_input 可用 = 样本在 mut 矩阵 AND CNA 矩阵 AND response 中都存在
  - 缺 mutation 或缺 CNA 的样本会被 concat join='inner' 自动排除，不会补 0

用法:
    conda run -n bnet_env python scripts/define_experiment_sets.py \
        --biolord_dir data/biolord_inputs \
        --output_dir data/biolord_inputs
"""

import os
import sys
import csv
import logging
import argparse
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 数据路径
PRAD_BASE_DIR      = "data/structured/prad"
PRAD_MUT_FILE      = os.path.join(PRAD_BASE_DIR, "processed", "final_analysis_set_cross_important_only.csv")
PRAD_CNA_FILE      = os.path.join(PRAD_BASE_DIR, "processed", "data_CNA_paper.csv")
PRAD_RESPONSE_FILE = os.path.join(PRAD_BASE_DIR, "processed", "response_paper.csv")
PRAD_SPLITS_DIR    = os.path.join(PRAD_BASE_DIR, "splits")

BRCA_BASE_DIR      = "data/structured/brca"
BRCA_MUT_FILE      = os.path.join(BRCA_BASE_DIR, "processed", "final_analysis_set_cross_important_only.csv")
BRCA_CNA_FILE      = os.path.join(BRCA_BASE_DIR, "processed", "data_CNA_paper.csv")
BRCA_RESPONSE_FILE = os.path.join(BRCA_BASE_DIR, "processed", "response_paper.csv")
BRCA_SPLITS_DIR    = os.path.join(BRCA_BASE_DIR, "splits")


def read_id_set(path: str, id_col: int = 0) -> set:
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return ids
        preferred = ["patient_id", "id", "sample_id"]
        key = next((name for name in preferred if name in reader.fieldnames), None)
        if key is None:
            usable = [name for name in reader.fieldnames if not name.lower().startswith("unnamed")]
            if not usable:
                return ids
            key = usable[id_col] if id_col < len(usable) else usable[0]
        for row in reader:
            value = str(row.get(key, "")).strip()
            if value:
                ids.add(value)
    return ids


def read_split_map(splits_dir: str) -> dict:
    """返回 {sample_id: split_name}。"""
    split_map = {}
    for split_name, fname in [("train", "training_set.csv"),
                               ("val",   "validation_set.csv"),
                               ("test",  "test_set.csv")]:
        path = os.path.join(splits_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"Split file not found: {path}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            key = next((name for name in ("id", "patient_id", "sample_id")
                        if name in reader.fieldnames), None)
            if key is None:
                usable = [name for name in reader.fieldnames
                          if not name.lower().startswith("unnamed") and name != "response"]
                if not usable:
                    raise ValueError(f"Cannot identify sample-ID column in {path}")
                key = usable[0]
            for row in reader:
                patient_id = str(row.get(key, "")).strip()
                if patient_id:
                    if patient_id in split_map:
                        raise ValueError(f"Sample {patient_id} occurs in multiple split files")
                    split_map[patient_id] = split_name
    return split_map


def read_response_map(response_file: str) -> dict:
    resp = {}
    if not os.path.exists(response_file):
        return resp
    with open(response_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return resp
        id_key = next((name for name in ("id", "patient_id", "sample_id")
                       if name in reader.fieldnames), None)
        label_key = next((name for name in ("response", "label")
                          if name in reader.fieldnames), None)
        if id_key is None or label_key is None:
            raise ValueError(
                f"{response_file} must contain an ID column and response/label column"
            )
        for row in reader:
            patient_id = str(row.get(id_key, "")).strip()
            label = str(row.get(label_key, "")).strip()
            if patient_id and label:
                resp[patient_id] = label
    return resp


def read_summary_ids(biolord_dir: str, cancer_type: str) -> set:
    subdir = "prad" if cancer_type == "prostate" else "brca"
    path = os.path.join(biolord_dir, subdir, "patient_summary.csv")
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("summary_text", "").strip()
            if text:
                ids.add(row["patient_id"].strip())
    return ids


def process_cancer(cancer_type: str, biolord_dir: str, output_dir: str, lines: list):
    """处理单个癌种，生成三类集合文件。"""

    def log(msg="", level="INFO"):
        if level == "ERROR":     logger.error(msg)
        elif level == "WARNING": logger.warning(msg)
        else:                    logger.info(msg)
        lines.append(msg)

    log(f"\n{'='*70}")
    log(f"  CANCER TYPE: {cancer_type.upper()}")
    log(f"{'='*70}")

    # 路径配置
    if cancer_type == "prostate":
        mut_file      = PRAD_MUT_FILE
        cna_file      = PRAD_CNA_FILE
        response_file = PRAD_RESPONSE_FILE
        splits_dir    = PRAD_SPLITS_DIR
    else:
        mut_file      = BRCA_MUT_FILE
        cna_file      = BRCA_CNA_FILE
        response_file = BRCA_RESPONSE_FILE
        splits_dir    = BRCA_SPLITS_DIR

    subdir = "prad" if cancer_type == "prostate" else "brca"
    out_dir = os.path.join(output_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # 读取各数据集
    mut_ids      = read_id_set(mut_file)
    cna_ids      = read_id_set(cna_file)
    resp_map     = read_response_map(response_file)
    resp_ids     = set(resp_map.keys())
    split_map    = read_split_map(splits_dir)
    split_ids    = set(split_map.keys())
    summary_ids  = read_summary_ids(biolord_dir, cancer_type)

    log(f"\n  Data availability:")
    log(f"    mut_ids      : {len(mut_ids)}")
    log(f"    cna_ids      : {len(cna_ids)}")
    log(f"    resp_ids     : {len(resp_ids)}")
    log(f"    split_ids    : {len(split_ids)}")
    log(f"    summary_ids  : {len(summary_ids)}")

    # 原模型真实可训练集合：mut & cna & resp（concat join='inner'）
    structured_ids = mut_ids & cna_ids & resp_ids
    log(f"\n  Original model trainable (mut & cna & resp): {len(structured_ids)}")
    log(f"  Excluded by missing mut : {len(resp_ids - mut_ids)}")
    log(f"  Excluded by missing cna : {len(resp_ids - cna_ids)}")
    log(f"  Excluded by missing both: {len(resp_ids - mut_ids - cna_ids)}")

    # 所有患者 = split_ids（以 split 为基准）
    all_patients = split_ids

    # -----------------------------------------------------------------------
    # 1. llm_text_set：有 summary_text + label + split
    # -----------------------------------------------------------------------
    llm_rows = []
    for pid in sorted(all_patients):
        has_text  = pid in summary_ids
        has_label = pid in resp_ids
        has_split = pid in split_map
        in_set = has_text and has_label and has_split
        label_val = resp_map.get(pid, "")
        try:
            label_int = int(float(label_val)) if label_val else ""
        except (ValueError, TypeError):
            label_int = ""
        llm_rows.append({
            "patient_id":      pid,
            "cancer_type":     cancer_type,
            "label":           label_int,
            "split":           split_map.get(pid, ""),
            "has_text_input":  int(has_text),
            "in_llm_text_set": int(in_set),
        })

    llm_path = os.path.join(out_dir, "llm_text_set.csv")
    with open(llm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "cancer_type", "label",
                                               "split", "has_text_input", "in_llm_text_set"])
        writer.writeheader()
        writer.writerows(llm_rows)
    n_llm = sum(1 for r in llm_rows if r["in_llm_text_set"] == 1)
    log(f"\n  llm_text_set: {n_llm} / {len(llm_rows)} -> {llm_path}")

    # -----------------------------------------------------------------------
    # 2. structured_set：原模型真实可训练（mut & cna & resp & split）
    # -----------------------------------------------------------------------
    struct_rows = []
    for pid in sorted(all_patients):
        has_mut   = pid in mut_ids
        has_cna   = pid in cna_ids
        has_label = pid in resp_ids
        has_split = pid in split_map
        # 原模型逻辑：必须同时有 mut 和 cna（concat join='inner'）
        in_set = has_mut and has_cna and has_label and has_split
        reason_parts = []
        if not has_mut:   reason_parts.append("missing_mutation")
        if not has_cna:   reason_parts.append("missing_cna")
        if not has_label: reason_parts.append("missing_label")
        reason = ";".join(reason_parts) if reason_parts else "ok"
        label_val = resp_map.get(pid, "")
        try:
            label_int = int(float(label_val)) if label_val else ""
        except (ValueError, TypeError):
            label_int = ""
        struct_rows.append({
            "patient_id":            pid,
            "cancer_type":           cancer_type,
            "label":                 label_int,
            "split":                 split_map.get(pid, ""),
            "has_structured_input":  int(has_mut and has_cna),
            "structured_input_reason": reason,
            "in_structured_set":     int(in_set),
        })

    struct_path = os.path.join(out_dir, "structured_set.csv")
    with open(struct_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "cancer_type", "label", "split",
                                               "has_structured_input", "structured_input_reason",
                                               "in_structured_set"])
        writer.writeheader()
        writer.writerows(struct_rows)
    n_struct = sum(1 for r in struct_rows if r["in_structured_set"] == 1)
    log(f"  structured_set: {n_struct} / {len(struct_rows)} -> {struct_path}")

    # -----------------------------------------------------------------------
    # 3. dual_strict_set：结构 & 文本 & label & split 全部可用
    # -----------------------------------------------------------------------
    dual_rows = []
    for pid in sorted(all_patients):
        has_mut   = pid in mut_ids
        has_cna   = pid in cna_ids
        has_text  = pid in summary_ids
        has_label = pid in resp_ids
        has_split = pid in split_map
        has_struct = has_mut and has_cna
        in_set = has_struct and has_text and has_label and has_split
        excl_parts = []
        if not has_mut:   excl_parts.append("missing_mutation")
        if not has_cna:   excl_parts.append("missing_cna")
        if not has_text:  excl_parts.append("missing_text")
        if not has_label: excl_parts.append("missing_label")
        excl_reason = ";".join(excl_parts) if excl_parts else ""
        label_val = resp_map.get(pid, "")
        try:
            label_int = int(float(label_val)) if label_val else ""
        except (ValueError, TypeError):
            label_int = ""
        dual_rows.append({
            "patient_id":            pid,
            "cancer_type":           cancer_type,
            "label":                 label_int,
            "split":                 split_map.get(pid, ""),
            "has_structured_input":  int(has_struct),
            "has_text_input":        int(has_text),
            "in_dual_strict_set":    int(in_set),
            "exclusion_reason":      excl_reason,
        })

    dual_path = os.path.join(out_dir, "dual_strict_set.csv")
    with open(dual_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "cancer_type", "label", "split",
                                               "has_structured_input", "has_text_input",
                                               "in_dual_strict_set", "exclusion_reason"])
        writer.writeheader()
        writer.writerows(dual_rows)
    n_dual = sum(1 for r in dual_rows if r["in_dual_strict_set"] == 1)
    log(f"  dual_strict_set: {n_dual} / {len(dual_rows)} -> {dual_path}")

    # -----------------------------------------------------------------------
    # 统计报告
    # -----------------------------------------------------------------------
    log(f"\n  Summary for {cancer_type.upper()}:")
    log(f"    Total patients (in split): {len(all_patients)}")
    log(f"    llm_text_set             : {n_llm}")
    log(f"    structured_set           : {n_struct}")
    log(f"    dual_strict_set          : {n_dual}")
    log(f"    Excluded from dual_strict: {len(all_patients) - n_dual}")

    # 排除原因统计
    excl_reasons = defaultdict(int)
    for r in dual_rows:
        if r["in_dual_strict_set"] == 0:
            for reason in r["exclusion_reason"].split(";"):
                if reason:
                    excl_reasons[reason] += 1
    for reason, cnt in sorted(excl_reasons.items()):
        log(f"      {reason}: {cnt}")

    # split 分布
    log(f"\n  dual_strict_set split distribution:")
    for sp in ["train", "val", "test"]:
        cnt = sum(1 for r in dual_rows if r["in_dual_strict_set"] == 1 and r["split"] == sp)
        log(f"    {sp}: {cnt}")

    return n_llm, n_struct, n_dual, len(all_patients)


def main():
    parser = argparse.ArgumentParser(description="Define experiment sample sets")
    parser.add_argument("--biolord_dir", default="data/biolord_inputs")
    parser.add_argument("--output_dir",  default="data/biolord_inputs")
    args = parser.parse_args()

    lines = []

    def log(msg=""):
        logger.info(msg)
        lines.append(msg)

    log("=" * 70)
    log("  EXPERIMENT SET DEFINITION REPORT")
    log("=" * 70)
    log("\nOriginal model structural input logic (from src/data/dataset.py):")
    log("  data_type = ['mut_important', 'cnv_del', 'cnv_amp']")
    log("  combine_type = 'union' (gene columns: union; sample rows: intersection)")
    log("  combine(): pd.concat(..., join='inner', axis=1)")
    log("    => samples must exist in BOTH mut_important AND CNA matrices")
    log("  load_data(): data.join(labels, how='inner')")
    log("    => samples must have a row in the data matrix")
    log("  Conclusion: structured_input_available = (in_mut AND in_cna AND in_response)")
    log("  Missing mutation OR missing CNA => automatically excluded by join='inner'")
    log("  No zero-filling for missing samples (only missing genes get fillna(0))")

    results = {}
    for ct in ["prostate", "breast"]:
        n_llm, n_struct, n_dual, n_total = process_cancer(
            ct, args.biolord_dir, args.output_dir, lines
        )
        results[ct] = (n_total, n_llm, n_struct, n_dual)

    log(f"\n{'='*70}")
    log("  OVERALL SUMMARY")
    log(f"{'='*70}")
    for ct, (n_total, n_llm, n_struct, n_dual) in results.items():
        log(f"  {ct.upper()}: total={n_total}, llm_text={n_llm}, structured={n_struct}, dual_strict={n_dual}")

    # 保存报告
    report_path = os.path.join(args.output_dir, "set_definition_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
