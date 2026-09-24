#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_dual_branch.py

End-to-end runner for the two independently trained KEPPNet branches.
The fusion weight is selected on the validation set by AUC; the classification
threshold remains fixed at 0.5 for validation and test evaluation.

用法:
    conda run -n biolord_gpu310 python run_dual_branch.py

Main configurations:
    PRAD: PathProNet(nl=4, lr=0.001, hub=30) + BioLORD(hd=128, dr=0.5, lr=0.0003)
    BRCA: PathProNet(nl=3, lr=0.001, hub=30) + BioLORD(hd=128, dr=0.3, lr=0.001)
"""

import os
import sys
import csv
import time
import numpy as np
import pandas as pd

import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

from logging import getLogger
from src.data.dataload import BnetDataLoader
from src.data.dataset import BnetData, set_cached_data_to_empty
from src.config.configuration import Config
from src.models.pathpronet import Model
from src.utils.logger import init_logger
from src.utils.utils import init_seed
from src.training.trainer import Trainer, model_predict

# ── 最优配置 ─────────────────────────────────────────────────────────────────
BEST_CONFIGS = {
    "prostate": {
        "study_names":  ["prad_p1000"],
        "model_name":   "prad-main",
        "dataset_name": "prostate_kg",
        "pathpronet": {
            "n_hidden_layers": 4, "lr": 0.001, "penalty": 0.001,
            "batch_size": 50, "hub_threshold": 30, "max_hub_proteins": 2,
            "max_mediators_per_edge": 10,
        },
        "biolord": {
            "hidden_dim": 128, "dropout": 0.5, "lr": 0.0003,
            "weight_decay": 0.0, "batch_size": 32, "seed": 2024,
            "pos_weight_multiplier": 1.0,
        },
    },
    "breast": {
        "study_names":  [
            "brca_igr_2015",
            "brca_mbcproject_wagle_2017",
            "brca_mbcproject_2022",
            "brca_tcga_pan_can_atlas_2018",
        ],
        "model_name":   "brca-main",
        "dataset_name": "breast_kg",
        "pathpronet": {
            "n_hidden_layers": 3, "lr": 0.001, "penalty": 0.001,
            "batch_size": 50, "hub_threshold": 30, "max_hub_proteins": 2,
            "max_mediators_per_edge": 10,
        },
        "biolord": {
            "hidden_dim": 128, "dropout": 0.3, "lr": 0.001,
            "weight_decay": 0.0001, "batch_size": 32, "seed": 2024,
            "pos_weight_multiplier": 1.0,
        },
    },
}

BIOLORD_DIR   = "data/biolord_inputs"


def metrics6(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, average_precision_score)
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"), 4),
        "auprc":     round(average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"), 4),
    }


def load_strict_ids(cancer_type):
    subdir = "prad" if cancer_type == "prostate" else "brca"
    ids = {}
    with open(os.path.join(BIOLORD_DIR, subdir, "dual_strict_set.csv")) as f:
        for row in csv.DictReader(f):
            if str(row.get("in_dual_strict_set", "0")).strip() == "1":
                ids[row["patient_id"].strip()] = {"label": row["label"], "split": row["split"]}
    return ids


def load_emb(cancer_type):
    subdir = "prad" if cancer_type == "prostate" else "brca"
    emb = np.load(os.path.join(BIOLORD_DIR, subdir, "embeddings.npy"))
    ids = []
    with open(os.path.join(BIOLORD_DIR, subdir, "embedding_patient_ids.csv")) as f:
        for row in csv.DictReader(f): ids.append(row["patient_id"].strip())
    meta = {}
    with open(os.path.join(BIOLORD_DIR, subdir, "embedding_meta.csv")) as f:
        for row in csv.DictReader(f):
            lbl = row["label"].strip()
            try: lbl = int(float(lbl))
            except: lbl = None
            if lbl is not None:
                meta[row["patient_id"].strip()] = {"label": lbl, "split": row["split"].strip()}
    return emb, ids, meta


# ── Step 1: 训练 PathProNet ───────────────────────────────────────────────────

def train_pathpronet(cancer_type, cfg):
    import torch
    logger = getLogger()
    logger.info(f"[PathProNet] Training on {cancer_type.upper()} ...")

    seed = 2024
    model_name   = cfg["model_name"]
    dataset_name = cfg["dataset_name"]
    pp_cfg       = cfg["pathpronet"]
    nl = pp_cfg["n_hidden_layers"]
    overrides = {
        "random_seed": seed,
        "reproducibility": False,
        "save_res": False,
        "interpretability": False,
        "n_hidden_layers": nl,
        "lr": pp_cfg["lr"],
        "penalty": pp_cfg["penalty"],
        "batch_size": pp_cfg["batch_size"],
        "hub_threshold": pp_cfg["hub_threshold"],
        "max_hub_proteins": pp_cfg["max_hub_proteins"],
        "max_mediators_per_edge": pp_cfg["max_mediators_per_edge"],
        "dropout": [0.5] + [0.1] * (nl + 1),
        "loss_weights": [2, 7, 20, 54, 148, 400, 1000, 2500][:nl * 2],
    }

    set_cached_data_to_empty()
    init_seed(seed, False)
    config = Config(model_name, dataset_name, config_dict=overrides)
    init_logger(config)

    dataset    = BnetData(config)
    dataloader = BnetDataLoader(config, dataset)
    model      = Model(config, dataset)
    trainer    = Trainer(config, dataloader, model)
    trainer.fix()

    device = config["device"]
    model.to(device); model.eval()

    strict_map = load_strict_ids(cancer_type)
    val_ids  = {p for p in strict_map if strict_map[p]["split"] == "val"}
    test_ids = {p for p in strict_map if strict_map[p]["split"] == "test"}
    val_mask  = np.array([p in val_ids  for p in dataloader.info_validate_])
    test_mask = np.array([p in test_ids for p in dataloader.info_test_])

    def pred(x_all, mask):
        x_sub = x_all[mask]
        if len(x_sub) == 0: return np.array([])
        xt = torch.tensor(x_sub, dtype=torch.float32)
        return model_predict(model, xt, device).cpu().numpy().ravel()

    val_prob  = pred(dataloader.x_validate_, val_mask)
    test_prob = pred(dataloader.x_test_,     test_mask)
    val_true  = dataloader.y_validate_[val_mask].ravel().astype(int)
    test_true = dataloader.y_test_[test_mask].ravel().astype(int)
    val_ids_list  = [p for p in dataloader.info_validate_ if p in val_ids]
    test_ids_list = [p for p in dataloader.info_test_     if p in test_ids]

    set_cached_data_to_empty()
    logger.info(f"[PathProNet] Done. val_n={len(val_true)}, test_n={len(test_true)}")
    return val_prob, test_prob, val_true, test_true, val_ids_list, test_ids_list


# ── Step 2: 训练 BioLORD 分类头 ──────────────────────────────────────────────

def train_biolord_head(cancer_type, cfg):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score

    logger = getLogger()
    logger.info(f"[BioLORD] Training classification head on {cancer_type.upper()} ...")

    emb, ids, meta = load_emb(cancer_type)
    strict_map = load_strict_ids(cancer_type)
    id2idx = {pid: i for i, pid in enumerate(ids)}

    def get_split(sp):
        pids = sorted(
            p for p in strict_map
            if strict_map[p]["split"] == sp and p in id2idx and p in meta
        )
        if not pids: return np.zeros((0, emb.shape[1])), np.zeros(0, dtype=int), []
        idxs = [id2idx[p] for p in pids]
        return emb[idxs], np.array([meta[p]["label"] for p in pids], dtype=int), pids

    Xtr, ytr, _      = get_split("train")
    Xv,  yv,  ids_v  = get_split("val")
    Xte, yte, ids_te = get_split("test")

    bc = cfg["biolord"]
    seed = bc["seed"]
    torch.manual_seed(seed); np.random.seed(seed)

    sc = StandardScaler()
    Xtr2 = sc.fit_transform(Xtr).astype(np.float32)
    Xv2  = sc.transform(Xv).astype(np.float32)
    Xte2 = sc.transform(Xte).astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hd, dr = bc["hidden_dim"], bc["dropout"]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(Xtr2.shape[1], hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, 1)
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    model = MLP().to(device)
    pwm = bc["pos_weight_multiplier"]
    pw  = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1) * pwm], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.Adam(model.parameters(), lr=bc["lr"], weight_decay=bc["weight_decay"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr2), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=bc["batch_size"], shuffle=True
    )

    best_auprc, best_state, no_imp = -1, None, 0
    for _ in range(500):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.from_numpy(Xv2).to(device))).cpu().numpy()
        try:    va = average_precision_score(yv, pv)
        except: va = 0
        if va > best_auprc:
            best_auprc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 30: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob  = torch.sigmoid(model(torch.from_numpy(Xv2).to(device))).cpu().numpy()
        test_prob = torch.sigmoid(model(torch.from_numpy(Xte2).to(device))).cpu().numpy()

    logger.info(f"[BioLORD] Done. val_n={len(yv)}, test_n={len(yte)}")
    return val_prob, test_prob, yv, yte, ids_v, ids_te


# ── Step 3: Late fusion (validation AUC selection; fixed 0.5 threshold) ───────

def search_and_fuse(bio_val_prob, bio_val_true, bio_val_ids,
                    bio_test_prob, bio_test_true, bio_test_ids,
                    llm_val_prob, llm_val_true, llm_val_ids,
                    llm_test_prob, llm_test_true, llm_test_ids):
    from sklearn.metrics import roc_auc_score

    bv = {pid: (y, p) for pid, y, p in zip(bio_val_ids,  bio_val_true,  bio_val_prob)}
    lv = {pid: (y, p) for pid, y, p in zip(llm_val_ids,  llm_val_true,  llm_val_prob)}
    bt = {pid: (y, p) for pid, y, p in zip(bio_test_ids, bio_test_true, bio_test_prob)}
    lt = {pid: (y, p) for pid, y, p in zip(llm_test_ids, llm_test_true, llm_test_prob)}

    cv = sorted(set(bv) & set(lv))
    ct = sorted(set(bt) & set(lt))

    if any(bv[p][0] != lv[p][0] for p in cv):
        raise ValueError("Structured and semantic validation labels do not match")
    if any(bt[p][0] != lt[p][0] for p in ct):
        raise ValueError("Structured and semantic test labels do not match")

    yv   = np.array([bv[p][0] for p in cv]); pbv = np.array([bv[p][1] for p in cv]); plv = np.array([lv[p][1] for p in cv])
    yt   = np.array([bt[p][0] for p in ct]); pbt = np.array([bt[p][1] for p in ct]); plt_ = np.array([lt[p][1] for p in ct])

    best_lam, decision_threshold, best_val_auc = 0.5, 0.5, -1
    for lam in [round(x * 0.1, 1) for x in range(11)]:
        pf_v = lam * pbv + (1 - lam) * plv
        val_auc = roc_auc_score(yv, pf_v) if len(set(yv)) > 1 else float("nan")
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_lam = lam

    pf_t = best_lam * pbt + (1 - best_lam) * plt_
    pf_v = best_lam * pbv + (1 - best_lam) * plv
    test_m = metrics6(yt, (pf_t >= decision_threshold).astype(int), pf_t)
    val_m = metrics6(yv, (pf_v >= decision_threshold).astype(int), pf_v)

    logger = getLogger()
    logger.info(
        f"[Fusion] Best λ={best_lam} (val_auc={best_val_auc:.4f}); "
        f"decision threshold={decision_threshold:.1f}"
    )
    return val_m, test_m, best_lam, decision_threshold, len(ct)


# ── 主运行函数 ────────────────────────────────────────────────────────────────

def run_dual_branch(cancer_type):
    """
    在指定癌种上运行完整的双分支 late fusion 模型。
    返回包含 6 项指标的结果字典。
    """
    logger = getLogger()
    cfg = BEST_CONFIGS[cancer_type]
    t0 = time.time()

    logger.info(f"\n{'='*65}")
    logger.info(f"  Dual-Branch Late Fusion: {cancer_type.upper()}")
    logger.info(f"  PathProNet: {cfg['pathpronet']}")
    logger.info(f"  BioLORD:    {cfg['biolord']}")
    logger.info(f"{'='*65}")

    # Step 1: PathProNet
    (bio_val_prob, bio_test_prob,
     bio_val_true, bio_test_true,
     bio_val_ids,  bio_test_ids) = train_pathpronet(cancer_type, cfg)

    # Step 2: BioLORD
    (llm_val_prob, llm_test_prob,
     llm_val_true, llm_test_true,
     llm_val_ids,  llm_test_ids) = train_biolord_head(cancer_type, cfg)

    # Step 3: fusion weight selected on validation AUC; threshold fixed at 0.5.
    val_m, test_m, best_lam, decision_threshold, n_test = search_and_fuse(
        bio_val_prob,  bio_val_true,  bio_val_ids,
        bio_test_prob, bio_test_true, bio_test_ids,
        llm_val_prob,  llm_val_true,  llm_val_ids,
        llm_test_prob, llm_test_true, llm_test_ids,
    )

    elapsed = time.time() - t0

    # 输出结果
    logger.info(f"\n{'='*65}")
    logger.info(
        f"  RESULTS: {cancer_type.upper()}  "
        f"(λ={best_lam}, threshold={decision_threshold})"
    )
    logger.info(f"  Test patients: {n_test}")
    logger.info(f"{'='*65}")
    logger.info(f"  {'Metric':<12} {'Val':>8} {'Test':>8}")
    logger.info(f"  {'-'*30}")
    for k in ["accuracy", "precision", "recall", "f1", "auc", "auprc"]:
        logger.info(f"  {k:<12} {val_m[k]:>8.4f} {test_m[k]:>8.4f}")
    logger.info(f"  Total time: {elapsed/60:.1f} min")

    result = {
        "cancer_type":  cancer_type,
        "lambda":       best_lam,
        "decision_threshold": decision_threshold,
        "test_n":       n_test,
        "elapsed_min":  round(elapsed / 60, 1),
    }
    result.update({f"val_{k}": v for k, v in val_m.items()})
    result.update({f"test_{k}": v for k, v in test_m.items()})
    return result


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    all_results = []

    # ── 前列腺癌（PRAD）────────────────────────────────────────────────────
    # 数据已经准备好，直接运行
    res_prad = run_dual_branch("prostate")
    all_results.append(res_prad)

    # ── 乳腺癌（BRCA）──────────────────────────────────────────────────────
    # 如果 BRCA 的 processed 数据还没准备，取消下面的注释
    # study_names = [
    #     "brca_igr_2015",
    #     "brca_mbcproject_wagle_2017",
    #     "brca_mbcproject_2022",
    #     "brca_tcga_pan_can_atlas_2018",
    # ]
    # prepare_data(study_names)
    # split_data()
    res_brca = run_dual_branch("breast")
    all_results.append(res_brca)

    # ── 汇总输出 ────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    print("\n" + "="*65)
    print("  FINAL SUMMARY: Dual-Branch Late Fusion")
    print("="*65)
    print(results_df[["cancer_type", "lambda", "decision_threshold",
                       "test_accuracy", "test_precision", "test_recall",
                       "test_f1", "test_auc", "test_auprc"]].to_string(index=False))

    # 保存结果
    os.makedirs("results/final", exist_ok=True)
    results_df.to_csv("results/final/dual_branch_results.csv", index=False)
    print("\nSaved: results/final/dual_branch_results.csv")
