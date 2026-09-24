#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_feature_fusion.py

Feature-level fusion: h_fuse = concat(h_bio, h_llm) -> MLP -> prediction
h_bio 和 h_llm 都冻结（不做端到端联合微调），只训练融合头。

用法:
    conda run -n biolord_gpu310 python scripts/run_feature_fusion.py
"""
import os, sys, csv, logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

CONFIGS = {
    "prostate": {
        "h_bio_npy":  "data/fusion_features/prad/h_bio.npy",
        "h_bio_ids":  "data/fusion_features/prad/h_bio_patient_ids.csv",
        "h_bio_meta": "data/fusion_features/prad/h_bio_meta.csv",
        "h_llm_npy":  "data/biolord_inputs/prad/embeddings.npy",       # baseline embeddings
        "h_llm_ids":  "data/biolord_inputs/prad/embedding_patient_ids.csv",
        "h_llm_meta": "data/biolord_inputs/prad/embedding_meta.csv",
        "set_file":   "data/biolord_inputs/prad/dual_strict_set.csv",
    },
    "breast": {
        "h_bio_npy":  "data/fusion_features/brca/h_bio.npy",
        "h_bio_ids":  "data/fusion_features/brca/h_bio_patient_ids.csv",
        "h_bio_meta": "data/fusion_features/brca/h_bio_meta.csv",
        "h_llm_npy":  "data/biolord_inputs/brca/embeddings.npy",
        "h_llm_ids":  "data/biolord_inputs/brca/embedding_patient_ids.csv",
        "h_llm_meta": "data/biolord_inputs/brca/embedding_meta.csv",
        "set_file":   "data/biolord_inputs/brca/dual_strict_set.csv",
    },
}


def load_repr(npy_path, ids_path, meta_path):
    emb = np.load(npy_path)
    ids = []
    with open(ids_path) as f:
        for row in csv.DictReader(f):
            ids.append(row["patient_id"].strip())
    meta = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            lbl = row["label"].strip()
            try: lbl = int(float(lbl))
            except: lbl = None
            if lbl is not None:
                meta[row["patient_id"].strip()] = {"label": lbl, "split": row["split"].strip()}
    return emb, ids, meta


def load_strict_ids(set_file):
    ids = set()
    with open(set_file) as f:
        for row in csv.DictReader(f):
            if str(row.get("in_dual_strict_set","0")).strip() == "1":
                ids.add(row["patient_id"].strip())
    return ids


def build_fused_splits(h_bio, bio_ids, bio_meta, h_llm, llm_ids, llm_meta, strict_ids):
    """对齐 h_bio 和 h_llm，拼接后按 split 分组。"""
    bio_id2idx = {pid: i for i, pid in enumerate(bio_ids)}
    llm_id2idx = {pid: i for i, pid in enumerate(llm_ids)}

    # 只保留两个分支都有的 strict set 样本
    common = strict_ids & set(bio_id2idx.keys()) & set(llm_id2idx.keys())
    common = common & set(bio_meta.keys()) & set(llm_meta.keys())

    for pid in common:
        if bio_meta[pid]["label"] != llm_meta[pid]["label"]:
            raise ValueError(f"Label mismatch between branches for patient {pid}")
        if bio_meta[pid]["split"] != llm_meta[pid]["split"]:
            raise ValueError(f"Split mismatch between branches for patient {pid}")

    splits = {"train": [], "val": [], "test": []}
    for pid in sorted(common):
        sp = bio_meta[pid]["split"]
        splits[sp].append(pid)

    result = {}
    for sp, pids in splits.items():
        if not pids:
            result[sp] = (np.zeros((0, h_bio.shape[1]+h_llm.shape[1])), np.zeros(0,dtype=int), [])
            continue
        bio_idxs = [bio_id2idx[p] for p in pids]
        llm_idxs = [llm_id2idx[p] for p in pids]
        X_bio = h_bio[bio_idxs]
        X_llm = h_llm[llm_idxs]
        X = np.concatenate([X_bio, X_llm], axis=1)
        y = np.array([bio_meta[p]["label"] for p in pids], dtype=int)
        result[sp] = (X, y, pids)
    return result, len(common)


def metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, average_precision_score)
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else float("nan"), 4),
        "auprc":     round(average_precision_score(y_true, y_prob) if len(set(y_true))>1 else float("nan"), 4),
    }


def run_fusion_mlp(Xtr, ytr, Xv, yv, Xte, yte, in_dim):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    sc = StandardScaler()
    Xtr2 = sc.fit_transform(Xtr).astype(np.float32)
    Xv2  = sc.transform(Xv).astype(np.float32)
    Xte2 = sc.transform(Xte).astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2024)
    np.random.seed(2024)

    class FusionMLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    model = FusionMLP(in_dim).to(device)
    pos_w = torch.tensor([(len(ytr)-ytr.sum())/max(ytr.sum(),1)], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr2), torch.from_numpy(ytr.astype(np.float32))),
                        batch_size=64, shuffle=True)

    best_auc, best_state, no_imp = -1, None, 0
    for epoch in range(300):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.from_numpy(Xv2).to(device))).cpu().numpy()
        try: vauc = roc_auc_score(yv, pv)
        except: vauc = 0
        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 30: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(Xte2).to(device))).cpu().numpy()
    return metrics(yte, (prob >= 0.5).astype(int), prob), prob


def run_one(cancer_type, cfg, output_dir):
    logger.info(f"\n=== Feature Fusion: {cancer_type.upper()} ===")
    ct_short = "prad" if cancer_type == "prostate" else "brca"

    h_bio, bio_ids, bio_meta = load_repr(cfg["h_bio_npy"], cfg["h_bio_ids"], cfg["h_bio_meta"])
    h_llm, llm_ids, llm_meta = load_repr(cfg["h_llm_npy"], cfg["h_llm_ids"], cfg["h_llm_meta"])
    strict_ids = load_strict_ids(cfg["set_file"])

    logger.info(f"  h_bio shape: {h_bio.shape}, h_llm shape: {h_llm.shape}")

    splits, n_common = build_fused_splits(h_bio, bio_ids, bio_meta, h_llm, llm_ids, llm_meta, strict_ids)
    Xtr, ytr, _      = splits["train"]
    Xv,  yv,  _      = splits["val"]
    Xte, yte, ids_te = splits["test"]
    in_dim = Xtr.shape[1]

    logger.info(f"  Common patients: {n_common}, fused dim: {in_dim}")
    logger.info(f"  train={len(ytr)}, val={len(yv)}, test={len(yte)}")

    m, prob = run_fusion_mlp(Xtr, ytr, Xv, yv, Xte, yte, in_dim)
    logger.info(f"  Feature Fusion MLP: acc={m['accuracy']} f1={m['f1']} auc={m['auc']} auprc={m['auprc']}")

    # 保存 metrics
    os.makedirs(output_dir, exist_ok=True)
    row = {"model": "FeatureFusion_MLP", "set": "dual_strict_set", "cancer_type": cancer_type,
           "h_bio_dim": h_bio.shape[1], "h_llm_dim": h_llm.shape[1], "fused_dim": in_dim,
           "train_n": len(ytr), "val_n": len(yv), "test_n": len(yte)}
    row.update(m)
    metrics_path = os.path.join(output_dir, f"{ct_short}_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    logger.info(f"  Saved: {metrics_path}")

    # 保存预测概率（供互补性分析）
    prob_dir = os.path.join(output_dir, "probs", ct_short)
    os.makedirs(prob_dir, exist_ok=True)
    with open(os.path.join(prob_dir, "test_probs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id","true_label","fusion_prob","fusion_pred"])
        w.writeheader()
        for pid, y, p in zip(ids_te, yte, prob):
            w.writerow({"patient_id": pid, "true_label": int(y),
                        "fusion_prob": float(p), "fusion_pred": int(p>=0.5)})

    return m


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", choices=["prostate", "breast"], default=None,
                        help="Cohort to evaluate; omit to run both cohorts")
    parser.add_argument("--output_dir", default="results/feature_fusion")
    args = parser.parse_args()

    all_results = {}
    selected = ([args.cancer_type] if args.cancer_type else list(CONFIGS))
    for ct in selected:
        cfg = CONFIGS[ct]
        all_results[ct] = run_one(ct, cfg, args.output_dir)

    # 汇总对比
    logger.info("\n" + "="*60)
    logger.info("  FEATURE FUSION SUMMARY")
    logger.info("="*60)
    for ct, m in all_results.items():
        logger.info(f"  {ct.upper()}: acc={m['accuracy']} f1={m['f1']} auc={m['auc']} auprc={m['auprc']}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
