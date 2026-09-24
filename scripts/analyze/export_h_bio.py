#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/export_pathpronet_patient_representation.py

从训练好的 PathProNet 导出患者级机制表示 h_bio。

选择依据：
  PathProNet 的 forward 流程：
    input(27687) -> h0/Diagonal(9229) -> hidden[0](166) -> hidden[1](11095) -> hidden[2](30)
  
  h_bio 选择 hidden_layers 最后一层的输出（h3，30维 for PRAD / 37维 for BRCA）：
  - 这是最顶层的通路级表示（Reactome top-level pathways）
  - 维度最小（30/37），便于与 768 维 BioLORD embedding 融合
  - 与最终预测头直接相连，信息最浓缩
  - 不需要改动模型结构，只需 hook 中间层输出

用法:
    conda run -n bnet_env python scripts/export_pathpronet_patient_representation.py \
        --cancer_type prostate

    conda run -n bnet_env python scripts/export_pathpronet_patient_representation.py \
        --cancer_type breast
"""
import os, sys, csv, logging, argparse
import numpy as np

import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def load_strict_set(biolord_dir: str, cancer_type: str):
    subdir = "prad" if cancer_type == "prostate" else "brca"
    path = os.path.join(biolord_dir, subdir, "dual_strict_set.csv")
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if str(row.get("in_dual_strict_set","0")).strip() == "1":
                pid = row["patient_id"].strip()
                rows[pid] = {"label": row["label"], "split": row["split"]}
    return rows


def extract_h_bio(model, X_tensor, device):
    """
    提取 PathProNet 最后一个 hidden layer 的输出（h3，top-level pathway representation）。
    使用 forward hook 捕获中间层激活。
    """
    import torch
    activations = {}

    def hook_fn(module, input, output):
        activations["h_bio"] = output.detach().cpu()

    # 注册 hook 到最后一个 hidden layer
    last_layer = model.hidden_layers[-1]
    handle = last_layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        # 分批处理避免 OOM
        batch_size = 256
        all_h = []
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i+batch_size].to(device)
            _ = model(xb)
            all_h.append(activations["h_bio"].numpy())
        h_bio = np.concatenate(all_h, axis=0)

    handle.remove()
    return h_bio


def run(cancer_type: str, biolord_dir: str, output_dir: str):
    from src.config.configuration import Config
    from src.data.dataset import BnetData, set_cached_data_to_empty
    from src.data.dataload import BnetDataLoader
    from src.models.pathpronet import Model
    from src.training.trainer import Trainer
    import torch

    if cancer_type == "prostate":
        model_name   = "prad-main"
        dataset_name = "prostate_kg"
    else:
        model_name   = "brca-main"
        dataset_name = "breast_kg"

    config = Config(model_name, dataset_name)
    config["save_res"] = False
    config["interpretability"] = False

    set_cached_data_to_empty()
    dataset    = BnetData(config)
    dataloader = BnetDataLoader(config, dataset)
    model      = Model(config, dataset)
    trainer    = Trainer(config, dataloader, model)

    logger.info(f"  Training PathProNet ({cancer_type}) ...")
    trainer.fix()

    device = config["device"]
    model.to(device)
    model.eval()

    logger.info(f"  Model h_bio layer: hidden_layers[-1] "
                f"({model.hidden_layers[-1].input_features} -> {model.hidden_layers[-1].output_features})")
    h_bio_dim = model.hidden_layers[-1].output_features

    # 读取 strict set
    strict_map = load_strict_set(biolord_dir, cancer_type)
    logger.info(f"  dual_strict_set: {len(strict_map)} patients")

    # 从 dataloader 中获取所有样本（train+val+test）
    all_x = np.concatenate([dataloader.x_train, dataloader.x_validate_, dataloader.x_test_], axis=0)
    all_info = list(dataloader.info_train) + list(dataloader.info_validate_) + list(dataloader.info_test_)
    all_y = np.concatenate([dataloader.y_train, dataloader.y_validate_, dataloader.y_test_], axis=0)

    # 只保留 strict set 中的样本
    strict_indices = [i for i, pid in enumerate(all_info) if pid in strict_map]
    strict_pids    = [all_info[i] for i in strict_indices]
    strict_x       = all_x[strict_indices]
    strict_y       = all_y[strict_indices].ravel()

    logger.info(f"  Extracting h_bio for {len(strict_pids)} patients ...")
    X_tensor = torch.tensor(strict_x, dtype=torch.float32)
    h_bio = extract_h_bio(model, X_tensor, device)
    logger.info(f"  h_bio shape: {h_bio.shape}")

    # 保存
    ct_short = "prad" if cancer_type == "prostate" else "brca"
    out_dir = os.path.join(output_dir, ct_short)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "h_bio.npy"), h_bio)

    with open(os.path.join(out_dir, "h_bio_patient_ids.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row_index", "patient_id"])
        for i, pid in enumerate(strict_pids): w.writerow([i, pid])

    with open(os.path.join(out_dir, "h_bio_meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_index","patient_id","label","split","cancer_type","h_bio_dim"])
        w.writeheader()
        for i, pid in enumerate(strict_pids):
            info = strict_map.get(pid, {})
            w.writerow({"row_index": i, "patient_id": pid,
                        "label": info.get("label",""), "split": info.get("split",""),
                        "cancer_type": cancer_type, "h_bio_dim": h_bio_dim})

    logger.info(f"  Saved h_bio -> {out_dir}")
    logger.info(f"  h_bio dim: {h_bio_dim} (top-level Reactome pathway representation)")

    set_cached_data_to_empty()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", required=True, choices=["prostate","breast"])
    parser.add_argument("--biolord_dir", default="data/biolord_inputs")
    parser.add_argument("--output_dir",  default="data/fusion_features")
    args = parser.parse_args()
    logger.info(f"=== Export h_bio: {args.cancer_type.upper()} ===")
    run(args.cancer_type, args.biolord_dir, args.output_dir)
    logger.info("Done.")
