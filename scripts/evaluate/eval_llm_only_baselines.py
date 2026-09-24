#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_llm_only_baselines.py

基于 BioLORD embeddings 跑 llm-only 基线（LR / Linear SVM / Small MLP）。
主结果基于 dual_strict_set，额外汇报 llm_text_set 上的结果。

用法:
    conda run -n biolord_gpu310 python scripts/run_llm_only_baselines.py \
        --cancer_type prostate --biolord_dir data/biolord_inputs --output_dir results/llm_only

    conda run -n biolord_gpu310 python scripts/run_llm_only_baselines.py \
        --cancer_type breast --biolord_dir data/biolord_inputs --output_dir results/llm_only
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


def load_embeddings_and_meta(biolord_dir: str, cancer_type: str):
    subdir = "prad" if cancer_type == "prostate" else "brca"
    emb_path  = os.path.join(biolord_dir, subdir, "embeddings.npy")
    ids_path  = os.path.join(biolord_dir, subdir, "embedding_patient_ids.csv")
    meta_path = os.path.join(biolord_dir, subdir, "embedding_meta.csv")

    emb = np.load(emb_path)
    ids = []
    with open(ids_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["patient_id"].strip())

    meta = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_str = row["label"].strip()
            try:
                label_int = int(float(label_str)) if label_str else None
            except (ValueError, TypeError):
                label_int = None
            if label_int is None:
                continue
            meta[row["patient_id"].strip()] = {
                "label": label_int,
                "split": row["split"].strip(),
            }

    assert len(ids) == emb.shape[0]
    logger.info(f"  Loaded embeddings: shape={emb.shape}, {len(ids)} patients")
    return emb, ids, meta


def load_set_ids(biolord_dir: str, cancer_type: str, set_name: str) -> set:
    """读取 llm_text_set / dual_strict_set 中 in_*=1 的 patient_id。"""
    subdir = "prad" if cancer_type == "prostate" else "brca"
    path = os.path.join(biolord_dir, subdir, f"{set_name}.csv")
    ids = set()
    if not os.path.exists(path):
        logger.warning(f"Set file not found: {path}")
        return ids
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 找 in_* 字段
            in_field = [k for k in row.keys() if k.startswith("in_")]
            if in_field and str(row[in_field[0]]).strip() == "1":
                ids.add(row["patient_id"].strip())
    return ids


def build_split_arrays(emb, ids, meta, allowed_ids):
    """按 split 构建 X/y 数组，只保留 allowed_ids 中的样本。"""
    id2idx = {pid: i for i, pid in enumerate(ids)}
    splits = {"train": [], "val": [], "test": []}
    for pid in sorted(allowed_ids):
        if pid not in id2idx or pid not in meta:
            continue
        splits[meta[pid]["split"]].append(pid)

    result = {}
    for sp, pids in splits.items():
        if not pids:
            result[sp] = (np.zeros((0, emb.shape[1])), np.zeros(0, dtype=int), [])
            continue
        idxs = [id2idx[p] for p in pids]
        X = emb[idxs]
        y = np.array([meta[p]["label"] for p in pids], dtype=int)
        result[sp] = (X, y, pids)
    return result


def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, average_precision_score)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc  = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "auprc": auprc}


def run_lr(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr, y_train)
    prob_val  = clf.predict_proba(X_v)[:, 1]
    prob_test = clf.predict_proba(X_te)[:, 1]
    pred_test = clf.predict(X_te)
    return (compute_metrics(y_test, pred_test, prob_test),
            prob_val, prob_test, clf.predict(X_v))


def run_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)
    X_te = scaler.transform(X_test)
    base = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(X_tr, y_train)
    prob_val  = clf.predict_proba(X_v)[:, 1]
    prob_test = clf.predict_proba(X_te)[:, 1]
    pred_test = clf.predict(X_te)
    return (compute_metrics(y_test, pred_test, prob_test),
            prob_val, prob_test, clf.predict(X_v))


def run_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_v  = scaler.transform(X_val).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)

    device = torch.device("cpu")
    torch.manual_seed(2024)
    np.random.seed(2024)

    class SmallMLP(nn.Module):
        def __init__(self, in_dim=768, hidden=256, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = SmallMLP(in_dim=X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # class weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    best_val_auc = -1
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(200):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # val AUC
        model.eval()
        with torch.no_grad():
            logits_v = model(torch.from_numpy(X_v).to(device)).cpu().numpy()
        prob_v = 1 / (1 + np.exp(-logits_v))
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(y_val, prob_v)
        except Exception:
            val_auc = 0.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_te = model(torch.from_numpy(X_te).to(device)).cpu().numpy()
        logits_v  = model(torch.from_numpy(X_v).to(device)).cpu().numpy()
    prob_test = 1 / (1 + np.exp(-logits_te))
    prob_val  = 1 / (1 + np.exp(-logits_v))
    pred_test = (prob_test >= 0.5).astype(int)
    pred_val  = (prob_val  >= 0.5).astype(int)
    return (compute_metrics(y_test, pred_test, prob_test),
            prob_val, prob_test, pred_val)


def save_metrics(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_for_set(emb, ids, meta, allowed_ids, set_label, cancer_type, output_dir):
    """在指定集合上跑三个模型，返回 metrics rows 和各模型的 val/test 概率。"""
    splits = build_split_arrays(emb, ids, meta, allowed_ids)
    X_tr, y_tr, ids_tr = splits["train"]
    X_v,  y_v,  ids_v  = splits["val"]
    X_te, y_te, ids_te = splits["test"]

    logger.info(f"  [{set_label}] train={len(y_tr)}, val={len(y_v)}, test={len(y_te)}")
    if len(y_tr) == 0 or len(y_te) == 0:
        logger.warning(f"  [{set_label}] Empty split, skipping")
        return [], {}

    metrics_rows = []
    probs = {}

    for model_name, run_fn in [("LR", run_lr), ("LinearSVM", run_svm), ("MLP", run_mlp)]:
        logger.info(f"  Running {model_name} on {set_label} ...")
        m, prob_val, prob_test, pred_val = run_fn(X_tr, y_tr, X_v, y_v, X_te, y_te)
        row = {"model": model_name, "set": set_label, "cancer_type": cancer_type,
               "train_n": len(y_tr), "val_n": len(y_v), "test_n": len(y_te)}
        row.update({k: round(v, 4) for k, v in m.items()})
        metrics_rows.append(row)
        probs[model_name] = {
            "val_ids": ids_v, "val_prob": prob_val, "val_true": y_v,
            "test_ids": ids_te, "test_prob": prob_test, "test_true": y_te,
        }
        logger.info(f"    {model_name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                    f"auc={m['auc']:.4f} auprc={m['auprc']:.4f}")

    return metrics_rows, probs


def main():
    parser = argparse.ArgumentParser(description="Run LLM-only baselines")
    parser.add_argument("--cancer_type", required=True, choices=["prostate", "breast"])
    parser.add_argument("--biolord_dir", default="data/biolord_inputs")
    parser.add_argument("--output_dir",  default="results/llm_only")
    args = parser.parse_args()

    ct = args.cancer_type
    ct_short = "prad" if ct == "prostate" else "brca"
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"=== LLM-only baselines: {ct.upper()} ===")

    # 加载 embeddings
    emb, ids, meta = load_embeddings_and_meta(args.biolord_dir, ct)

    # 加载集合
    dual_ids = load_set_ids(args.biolord_dir, ct, "dual_strict_set")
    llm_ids  = load_set_ids(args.biolord_dir, ct, "llm_text_set")
    logger.info(f"  dual_strict_set: {len(dual_ids)}, llm_text_set: {len(llm_ids)}")

    all_rows = []
    all_probs = {}

    # 主结果：dual_strict_set
    rows_dual, probs_dual = run_for_set(
        emb, ids, meta, dual_ids, "dual_strict_set", ct, args.output_dir
    )
    all_rows.extend(rows_dual)
    all_probs["dual_strict_set"] = probs_dual

    # 补充：llm_text_set（不可与 dual_strict 直接横比）
    rows_llm, probs_llm = run_for_set(
        emb, ids, meta, llm_ids, "llm_text_set", ct, args.output_dir
    )
    all_rows.extend(rows_llm)
    all_probs["llm_text_set"] = probs_llm

    # 保存 metrics
    metrics_path = os.path.join(args.output_dir, f"{ct_short}_metrics.csv")
    save_metrics(all_rows, metrics_path)
    logger.info(f"  Metrics saved: {metrics_path}")

    # 保存 dual_strict 的预测概率（供 late fusion 使用）
    for model_name, prob_data in probs_dual.items():
        prob_dir = os.path.join(args.output_dir, "probs", ct_short)
        os.makedirs(prob_dir, exist_ok=True)
        # val probs
        val_rows = [{"patient_id": pid, "true_label": int(y), "llm_prob": float(p)}
                    for pid, y, p in zip(prob_data["val_ids"], prob_data["val_true"], prob_data["val_prob"])]
        with open(os.path.join(prob_dir, f"{model_name}_val_probs.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient_id", "true_label", "llm_prob"])
            writer.writeheader(); writer.writerows(val_rows)
        # test probs
        test_rows = [{"patient_id": pid, "true_label": int(y), "llm_prob": float(p)}
                     for pid, y, p in zip(prob_data["test_ids"], prob_data["test_true"], prob_data["test_prob"])]
        with open(os.path.join(prob_dir, f"{model_name}_test_probs.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient_id", "true_label", "llm_prob"])
            writer.writeheader(); writer.writerows(test_rows)

    logger.info(f"=== DONE: {ct.upper()} ===")


if __name__ == "__main__":
    main()
