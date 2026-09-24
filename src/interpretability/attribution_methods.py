"""
ProPathNet / reactome_protein_kg（pathway_protein_kg_enhanced）专用归因分析（重写版）

该模块专门服务你的“知识增强通路-蛋白混合网络”：
- 不假设“纯通路/纯蛋白”结构，完全以模型真实的 `feature_names / hidden_layers / linears` 为准
- 解释模型输出向量中的某一列（默认最后一列=最深层输出）
- 导出 `src/utils/sankey.py` 兼容的文件到 `${output_dir}/extracted/`：
  - gradient_importance_0.csv        (输入MultiIndex特征的归因，Sankey第一层用它)
  - link_weights_1..L.csv            (h0->h1, ..., hK->root)
  - node_importance_graph_adjusted.csv (每层节点重要性，含 layer/coef_combined)
  - topk_layer_*.csv + topk_summary.csv

配置项（可选）：
- interpret_output_index: int   # 解释哪一列输出（0..n_hidden_layers），默认 -1(最后一列)
- method_name / feature_important_name: 'deeplift' | 'integratedgradients'
- baseline: 'zero' | 'mean'
- topk_per_layer: int
- generate_sankey: bool

入口函数：interpret_propathnet_model(config, model, dataloader)
"""

from __future__ import annotations

import os
import copy
from os.path import join, exists
from logging import getLogger
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, tanh, sigmoid
from captum.attr import LayerDeepLift, LayerIntegratedGradients

from src.models.custom.layer_custom import Diagonal

try:
    from src.utils import sankey
    _SANKEY_AVAILABLE = True
except Exception:
    _SANKEY_AVAILABLE = False

try:
    # New tidy Sankey for hybrid KG network
    from src.utils import sankey_hybrid
    _SANKEY_HYBRID_AVAILABLE = True
except Exception:
    sankey_hybrid = None
    _SANKEY_HYBRID_AVAILABLE = False


def _cval(config, key: str, default=None):
    if config is None:
        return default
    try:
        v = config.get(key, default)  # dict-like
        if v is not None:
            return v
    except Exception:
        pass
    try:
        v = config[key]
        if v is not None:
            return v
    except Exception:
        pass
    if hasattr(config, key):
        v = getattr(config, key)
        if v is not None:
            return v
    try:
        v = config.__dict__.get(key, default)
        if v is not None:
            return v
    except Exception:
        pass
    return default


def _ensure_dir(path: str) -> None:
    if not exists(path):
        os.makedirs(path, exist_ok=True)


def _sorted_h_keys(feature_names: Dict[str, object]) -> List[str]:
    hs = [k for k in feature_names.keys() if k.startswith("h")]
    return sorted(hs, key=lambda x: int(x[1:]) if x[1:].isdigit() else 10**9)


def _pick_output_index(model: nn.Module, config) -> int:
    out_dim = int(getattr(model, "n_hidden_layers", 0)) + 1
    idx = _cval(config, "interpret_output_index", None)
    if idx is None:
        return out_dim - 1
    idx = int(idx)
    if idx < 0:
        idx = out_dim + idx
    if idx < 0 or idx >= out_dim:
        raise ValueError(f"interpret_output_index={idx} out of range (0..{out_dim-1})")
    return idx


class _ProPathNetScalarSubModel(nn.Module):
    """
    Captum 用子模型：输出一个标量（指定 output_index 对应的那一列输出）。
    """

    def __init__(self, model: nn.Module, output_index: int):
        super().__init__()
        self.output_index = int(output_index)

        # dropout
        self.dropout1 = copy.deepcopy(model.dropout1)
        self.dropout2 = copy.deepcopy(model.dropout2)

        # diagonal layer (inputs -> genes)
        self.h0_diag = copy.deepcopy(model.h0)

        # optional GCN
        self.use_gcn = bool(getattr(model, "gcn", False))
        if self.use_gcn:
            self.g0 = copy.deepcopy(model.g0)
            self.g1 = copy.deepcopy(model.g1)
            self.gene_adj_matrix = copy.deepcopy(model.gene_adj_matrix)
        else:
            self.g0 = None
            self.g1 = None
            self.gene_adj_matrix = None

        # hidden layers (h{i} -> h{i+1})
        self.hidden_layers = nn.ModuleList([copy.deepcopy(m) for m in model.hidden_layers])

        # the readout for the selected output index
        self.output_linear = copy.deepcopy(model.linears[self.output_index])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = tanh(self.h0_diag(x))

        if self.use_gcn and self.gene_adj_matrix is not None:
            out = self.g0(out, self.gene_adj_matrix)
            out = self.dropout1(out)
            out = self.g1(out, self.gene_adj_matrix)
            out = self.dropout1(out)
        else:
            out = self.dropout1(out)

        # apply prefix layers up to output_index
        for i in range(self.output_index):
            out = tanh(self.hidden_layers[i](out))
            out = self.dropout2(out)

        y = sigmoid(self.output_linear(out))
        return y  # [N, 1]


class ProPathNetAttributor:
    def __init__(self, config, model: nn.Module, output_dir: str):
        self.config = config
        self.model = model
        self.output_dir = output_dir
        self.logger = getLogger()

        self.extracted_dir = join(self.output_dir, "extracted")
        _ensure_dir(self.extracted_dir)

        # method / baseline
        self.method_name = _cval(config, "feature_important_name", None) or _cval(config, "method_name", "deeplift")
        self.baseline_name = _cval(config, "baseline", "zero")
        self.topk = int(_cval(config, "topk_per_layer", 10) or 10)
        self.generate_sankey = bool(_cval(config, "generate_sankey", True))

        self.output_index = _pick_output_index(model, config)

        # ordered layer keys
        self.h_keys = _sorted_h_keys(self.model.feature_names)
        self.layer_keys = ["inputs"] + self.h_keys

        # sankey expects n_layer including root
        self.n_layer_for_sankey = len(self.layer_keys) + 1

    def _build_baseline(self, X: torch.Tensor) -> Union[int, torch.Tensor]:
        if str(self.baseline_name).lower() == "mean":
            return torch.mean(X, dim=0, keepdim=True)
        return 0

    def _make_attributor(self, sub_model: nn.Module, layer: nn.Module):
        if str(self.method_name).lower() in ("deeplift", "deep_lift"):
            return LayerDeepLift(sub_model, layer)
        return LayerIntegratedGradients(sub_model, layer)

    def compute_node_importance(self, X: Union[np.ndarray, torch.Tensor]) -> OrderedDict[str, pd.DataFrame]:
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        baseline = self._build_baseline(X)

        sub_model = _ProPathNetScalarSubModel(self.model, self.output_index)
        sub_model.eval()

        node_weights: OrderedDict[str, pd.DataFrame] = OrderedDict()

        # 0) inputs attribution (saved as gradient_importance_0.csv)
        attributor = self._make_attributor(sub_model, sub_model.h0_diag)
        contrib = attributor.attribute(X, baselines=baseline, attribute_to_layer_input=True)
        w = np.sum(contrib.detach().cpu().numpy(), axis=0)
        w = np.abs(w).reshape(-1)
        idx = self.model.feature_names["inputs"]
        if len(w) != len(idx):
            m = min(len(w), len(idx))
            w = w[:m]
            idx = idx[:m]
            self.logger.warning(f"inputs attribution length mismatch, trimmed to {m}")
        node_weights["inputs"] = pd.DataFrame(w, index=idx, columns=["coef"])

        # 1) h* node importances
        # We attribute to hidden_layers[i] input => corresponds to h{i} nodes.
        for hk in self.h_keys:
            layer_idx = int(hk[1:]) if hk[1:].isdigit() else None
            if layer_idx is None:
                continue
            feats = self.model.feature_names[hk]

            if layer_idx < len(sub_model.hidden_layers):
                layer = sub_model.hidden_layers[layer_idx]
            else:
                # last hK: use output linear input attribution as approximation
                layer = sub_model.output_linear

            attributor = self._make_attributor(sub_model, layer)
            contrib = attributor.attribute(X, baselines=baseline, attribute_to_layer_input=True)
            w = np.sum(contrib.detach().cpu().numpy(), axis=0)
            w = np.abs(w).reshape(-1)
            if len(w) != len(feats):
                m = min(len(w), len(feats))
                w = w[:m]
                feats = feats[:m]
                self.logger.warning(f"{hk} attribution length mismatch, trimmed to {m}")
            node_weights[hk] = pd.DataFrame(w, index=feats, columns=["coef"]).sort_values("coef", ascending=False)

        return node_weights

    def _layer_weight_matrix_in_out(self, layer: nn.Module) -> np.ndarray:
        """
        Return weights in shape (n_in, n_out) with mask applied (source rows, target cols).
        """
        if isinstance(layer, Diagonal):
            w = layer.weight.detach()
            if hasattr(layer, "mask") and isinstance(layer.mask, torch.Tensor):
                w = torch.where(layer.mask.data == 1, w, torch.zeros_like(w))
            return w.t().cpu().numpy()

        w = layer.weight.detach()
        if hasattr(layer, "mask") and isinstance(layer.mask, torch.Tensor):
            w = torch.where(layer.mask.data == 1, w, torch.zeros_like(w))
        return w.t().cpu().numpy()

    def compute_link_weights(self) -> Dict[str, pd.DataFrame]:
        """
        Build:
          link_weights_1:  h0 -> h1   (hidden_layers[0])
          ...
          link_weights_K:  h{K-1} -> hK
          link_weights_{K+1}: hK -> root  (selected output linear)
        """
        sub_model = _ProPathNetScalarSubModel(self.model, self.output_index)
        sub_model.eval()

        link_weights: Dict[str, pd.DataFrame] = OrderedDict()

        # hidden layer connections
        for i in range(len(sub_model.hidden_layers)):
            src_key = f"h{i}"
            tgt_key = f"h{i+1}"
            if src_key not in self.model.feature_names or tgt_key not in self.model.feature_names:
                continue
            src = self.model.feature_names[src_key]
            tgt = self.model.feature_names[tgt_key]
            w = self._layer_weight_matrix_in_out(sub_model.hidden_layers[i])  # (|src|, |tgt|)
            n_in = min(w.shape[0], len(src))
            n_out = min(w.shape[1], len(tgt))
            link_weights[f"{src_key}_to_{tgt_key}"] = pd.DataFrame(w[:n_in, :n_out], index=src[:n_in], columns=tgt[:n_out])

        # last to root (only export for the chosen output index)
        last_h = f"h{self.output_index}"
        if last_h in self.model.feature_names:
            src = self.model.feature_names[last_h]
            w = sub_model.output_linear.weight.detach().t().cpu().numpy()  # (|src|, 1)
            n_in = min(w.shape[0], len(src))
            link_weights[f"{last_h}_to_root"] = pd.DataFrame(w[:n_in, :], index=src[:n_in], columns=["root"])

        return link_weights

    def export_files(self, node_weights: OrderedDict[str, pd.DataFrame], link_weights: Dict[str, pd.DataFrame]) -> None:
        # gradient_importance_*.csv
        for i, (k, df) in enumerate(node_weights.items()):
            df.to_csv(join(self.extracted_dir, f"gradient_importance_{i}.csv"))

        # link_weights_*.csv in sankey expected order
        lw_list: List[pd.DataFrame] = []
        for i in range(self.output_index):
            key = f"h{i}_to_h{i+1}"
            if key in link_weights:
                lw_list.append(link_weights[key])
        root_key = f"h{self.output_index}_to_root"
        if root_key in link_weights:
            lw_list.append(link_weights[root_key])
        for i, df in enumerate(lw_list, start=1):
            df.to_csv(join(self.extracted_dir, f"link_weights_{i}.csv"))

        # node_importance_graph_adjusted.csv (for sankey)
        rows = []
        for hk in self.h_keys:
            if hk not in node_weights:
                continue
            layer_num = int(hk[1:]) + 1  # h0 => layer 1
            df = node_weights[hk]
            for node_id, r in df.iterrows():
                rows.append(
                    {
                        "node_id": str(node_id),
                        "coef": float(r["coef"]),
                        "coef_combined": float(r["coef"]),
                        "layer": layer_num,
                    }
                )
        node_imp = pd.DataFrame(rows).set_index("node_id") if rows else pd.DataFrame(columns=["coef", "coef_combined", "layer"])
        node_imp.to_csv(join(self.extracted_dir, "node_importance_graph_adjusted.csv"))

        # top-k per layer
        summary = []
        for hk in self.h_keys:
            if hk not in node_weights:
                continue
            layer_num = int(hk[1:]) + 1
            df = node_weights[hk].copy()
            df["_score_"] = df["coef"].abs()
            top = df.sort_values("_score_", ascending=False).head(self.topk)[["coef"]].rename(columns={"coef": "score"})
            top["rank"] = range(1, len(top) + 1)
            top.to_csv(join(self.extracted_dir, f"topk_layer_{layer_num}.csv"))
            for node_id, r in top.iterrows():
                summary.append({"layer": layer_num, "node": str(node_id), "score": float(r["score"]), "rank": int(r["rank"])})
        if summary:
            pd.DataFrame(summary).sort_values(["layer", "rank"]).to_csv(join(self.extracted_dir, "topk_summary.csv"), index=False)

    def run_sankey(self, node_weights: OrderedDict[str, pd.DataFrame], link_weights: Dict[str, pd.DataFrame]) -> None:
        if not self.generate_sankey:
            return

        # Prefer the new hybrid Sankey for KG-enhanced hybrid network.
        sankey_mode = str(_cval(self.config, "sankey_mode", "") or "").lower()
        network_flag = str(_cval(self.config, "network", "") or "").lower()
        use_hybrid = (sankey_mode == "hybrid") or (network_flag == "reactome_protein_kg")

        if use_hybrid and _SANKEY_HYBRID_AVAILABLE and sankey_hybrid is not None:
            try:
                topk = int(_cval(self.config, "topk_per_layer", None) or 10)
                flow_norm = str(_cval(self.config, "sankey_flow_normalization", "source") or "source").lower()
                sankey_hybrid.run(
                    output_dir=self.output_dir,
                    node_weights=node_weights,
                    link_weights=link_weights,
                    h_keys=self.h_keys,
                    topk_per_layer=topk,
                    flow_normalization=flow_norm,
                    title="KG-enhanced pathway–protein network (hybrid Sankey)",
                )
                return
            except Exception as e:
                self.logger.warning(f"hybrid sankey failed, falling back to legacy sankey: {e}")

        if not _SANKEY_AVAILABLE:
            self.logger.warning("legacy sankey module not available; skip")
            return
        sankey.run(self.output_dir, self.n_layer_for_sankey)

    def interpret(self, X: Union[np.ndarray, torch.Tensor]) -> Dict[str, object]:
        self.logger.info(
            f"interpret_propathnet_model: method={self.method_name}, baseline={self.baseline_name}, output_index={self.output_index}"
        )
        node_weights = self.compute_node_importance(X)
        link_weights = self.compute_link_weights()
        self.export_files(node_weights, link_weights)
        self.run_sankey(node_weights, link_weights)
        return {"node_weights": node_weights, "link_weights": link_weights, "output_index": self.output_index}


def interpret_propathnet_model(config, model: nn.Module, dataloader):
    logger = getLogger()
    if _cval(config, "interpretability", True) is False:
        logger.info("interpretability disabled, skip")
        return None

    output_dir = _cval(config, "output_dir", "results")

    X = dataloader.x_test_
    model.to("cpu")
    model.eval()

    runner = ProPathNetAttributor(config=config, model=model, output_dir=output_dir)
    return runner.interpret(X)


# alias for compatibility
interpret = interpret_propathnet_model

