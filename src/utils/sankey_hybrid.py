"""
Hybrid Sankey generator for ProPathNet / reactome_protein_kg.

Design goals (vs legacy `src/utils/sankey.py`):
- Deterministic, tidy layout (fixed x by layer, y by within-layer ranking)
- Correct top-k node selection that matches exported attributions
- Handles mixed layers (genes / pathways / proteins) without assuming alternating types
- Produces a single self-contained HTML (always), and tries PNG/PDF if Kaleido is available
"""

from __future__ import annotations

import os
import re
from os.path import join, exists
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


_RE_REACTOME = re.compile(r"^R-HSA-\d+$")
_RE_STRING_PROTEIN = re.compile(r"^9606\.ENSP\d+$")


def _ensure_dir(path: str) -> None:
    if not exists(path):
        os.makedirs(path, exist_ok=True)


def _safe_float(x) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _read_reactome_id_to_name() -> Dict[str, str]:
    path = join(os.getcwd(), "data/pathways/Reactome/ReactomePathways.txt")
    try:
        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = ["id", "name", "species"]
        df = df[df["species"] == "Homo sapiens"]
        return dict(zip(df["id"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def _read_pathway_short_names() -> Dict[str, str]:
    """
    Optional mapping: Full name -> short name.
    """
    xlsx = join(os.getcwd(), "data/pathways/pathways_short_names.xlsx")
    try:
        df = pd.read_excel(xlsx)
        if "Full name" not in df.columns:
            return {}
        short_col = None
        for c in df.columns:
            if "Short name" in str(c):
                short_col = c
                break
        if short_col is None:
            return {}
        m = {}
        for full, short in zip(df["Full name"].astype(str).values, df[short_col].astype(str).values):
            if short and short.lower() != "nan":
                m[full] = short
        return m
    except Exception:
        return {}


def _read_string_id_to_preferred_name() -> Dict[str, str]:
    path = join(os.getcwd(), "data/protein/9606.protein.info.v12.0.txt")
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        # columns: #string_protein_id, preferred_name, annotation, ...
        if "#string_protein_id" not in df.columns or "preferred_name" not in df.columns:
            return {}
        return dict(zip(df["#string_protein_id"].astype(str), df["preferred_name"].astype(str)))
    except Exception:
        return {}


def _infer_node_type(node_id: str) -> str:
    if node_id == "root":
        return "root"
    if _RE_REACTOME.match(node_id):
        return "pathway"
    if _RE_STRING_PROTEIN.match(node_id):
        return "protein"
    # common gene symbols: TP53, AR, BRCA1...
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{0,30}$", node_id):
        return "gene_or_symbol"
    return "other"


def _node_color(node_type: str) -> str:
    # visually distinct but calm palette
    if node_type == "root":
        return "rgba(40,40,40,0.9)"
    if node_type == "pathway":
        return "rgba(120,90,200,0.75)"
    if node_type == "protein":
        return "rgba(240,140,60,0.75)"
    if node_type == "gene_or_symbol":
        return "rgba(60,140,220,0.75)"
    return "rgba(200,200,200,0.65)"


def _edge_color(src_color: str, alpha: float = 0.25) -> str:
    # Convert rgba(r,g,b,a) -> rgba(r,g,b,alpha)
    m = re.match(r"rgba\((\d+),(\d+),(\d+),([0-9.]+)\)", src_color.replace(" ", ""))
    if not m:
        return f"rgba(160,160,160,{alpha})"
    r, g, b = m.group(1), m.group(2), m.group(3)
    return f"rgba({r},{g},{b},{alpha})"


def _label_for_node(
    node_id: str,
    reactome_id_to_name: Dict[str, str],
    reactome_short_name: Dict[str, str],
    string_id_to_name: Dict[str, str],
) -> str:
    if node_id.startswith("OTHER__"):
        return "Other"
    if node_id == "root":
        return "Outcome"
    if node_id in reactome_id_to_name:
        full = reactome_id_to_name[node_id]
        return reactome_short_name.get(full, full)
    if node_id in string_id_to_name:
        return string_id_to_name[node_id]
    return node_id


def _rank_map_for_layers(h_keys: List[str], node_weights: Dict[str, pd.DataFrame], topk: int) -> Dict[str, int]:
    """
    Build a mapping node_id -> rank (1..topk) per h-layer based strictly on attribution.
    If the same node_id appears in multiple layers (rare), the first occurrence wins.
    """
    m: Dict[str, int] = {}
    for hk in h_keys:
        top = _topk_from_node_weights(node_weights, hk, int(topk))
        for i, nid in enumerate(top, start=1):
            if nid not in m:
                m[str(nid)] = int(i)
    return m


def _label_with_rank(base_label: str, node_id: str, rank_map: Dict[str, int]) -> str:
    if node_id == "root" or str(node_id).startswith("OTHER__"):
        return base_label
    r = rank_map.get(str(node_id))
    if r is None:
        return base_label
    return f"{int(r)}. {base_label}"


def _topk_from_node_weights(node_weights: Dict[str, pd.DataFrame], hk: str, k: int) -> List[str]:
    """
    node_weights[hk] is a DF indexed by node id with column 'coef'.
    """
    if hk not in node_weights:
        return []
    df = node_weights[hk]
    if df is None or df.empty:
        return []
    # ensure numeric and stable
    s = df["coef"].apply(_safe_float).abs()
    return list(s.sort_values(ascending=False).head(int(k)).index.astype(str))


def _node_importance_series(node_weights: Dict[str, pd.DataFrame], hk: str) -> pd.Series:
    if hk not in node_weights or node_weights[hk] is None or node_weights[hk].empty:
        return pd.Series(dtype=float)
    s = node_weights[hk]["coef"].apply(_safe_float).abs()
    s.index = s.index.astype(str)
    return s


def _build_layer_nodes(
    h_keys: List[str],
    node_weights: Dict[str, pd.DataFrame],
    topk: int,
) -> Dict[str, List[str]]:
    """
    Returns: {hk: [node_id, ..., OTHER__hk]}.
    """
    ret: Dict[str, List[str]] = {}
    for hk in h_keys:
        top = _topk_from_node_weights(node_weights, hk, topk)
        other = f"OTHER__{hk}"
        ret[hk] = top + [other]
    return ret


def _augment_target_nodes_by_connectivity(
    w: pd.DataFrame,
    src_nodes: List[str],
    tgt_nodes: List[str],
    *,
    add_k: int = 5,
) -> List[str]:
    """
    Add up to add_k target nodes that have the largest total absolute incoming weight
    from the selected src_nodes.
    """
    if add_k <= 0:
        return tgt_nodes
    if w is None or w.empty:
        return tgt_nodes

    src_nodes = [str(s) for s in src_nodes if not str(s).startswith("OTHER__")]
    tgt_nodes = [str(t) for t in tgt_nodes if not str(t).startswith("OTHER__")]

    w2 = w.copy()
    w2.index = w2.index.astype(str)
    w2.columns = w2.columns.astype(str)

    src_exist = [s for s in src_nodes if s in w2.index]
    if not src_exist:
        return tgt_nodes

    # score each target by incoming mass from selected sources
    try:
        scores = w2.loc[src_exist].abs().sum(axis=0)
    except Exception:
        scores = w2.loc[src_exist].applymap(lambda x: abs(_safe_float(x))).sum(axis=0)

    scores = scores.sort_values(ascending=False)
    added: List[str] = []
    for t in scores.index:
        if t in tgt_nodes:
            continue
        added.append(str(t))
        if len(added) >= int(add_k):
            break
    return tgt_nodes + added


def _select_topk_connected(
    *,
    prev_nodes: List[str],
    w_prev_to_cur: Optional[pd.DataFrame],
    cur_importance: Optional[pd.Series],
    topk: int,
    candidate_pool: int = 50,
) -> List[str]:
    """
    Select exactly topk nodes for current layer that are both:
    - high attribution (cur_importance)
    - connected (non-zero incoming) from prev_nodes in weight matrix

    Strategy:
    1) Take candidate_pool highest by attribution
    2) Compute incoming mass from prev_nodes
    3) Rank by (incoming > 0) then incoming (desc) then attribution (desc)
    4) If still < topk, fill by attribution-only.
    """
    topk = int(topk)
    if topk <= 0:
        return []

    # attribution candidates
    if cur_importance is None or cur_importance.empty:
        candidates: List[str] = []
    else:
        s = cur_importance.copy()
        s.index = s.index.astype(str)
        s = s.apply(_safe_float).abs()
        candidates = list(s.sort_values(ascending=False).head(int(candidate_pool)).index.astype(str))

    # connectivity scores
    incoming = pd.Series(dtype=float)
    if w_prev_to_cur is not None and not w_prev_to_cur.empty and candidates:
        w = w_prev_to_cur.copy()
        w.index = w.index.astype(str)
        w.columns = w.columns.astype(str)
        prev = [str(n) for n in prev_nodes if (not str(n).startswith("OTHER__")) and (str(n) in w.index)]
        if prev:
            cols = [c for c in candidates if c in w.columns]
            if cols:
                try:
                    incoming = w.loc[prev, cols].abs().sum(axis=0)
                except Exception:
                    incoming = w.loc[prev, cols].applymap(lambda x: abs(_safe_float(x))).sum(axis=0)

    # build ranking
    if cur_importance is None or cur_importance.empty:
        imp = pd.Series(dtype=float)
    else:
        imp = cur_importance.copy()
        imp.index = imp.index.astype(str)
        imp = imp.apply(_safe_float).abs()

    rows = []
    for n in candidates:
        inc = float(incoming.get(n, 0.0)) if incoming is not None else 0.0
        rows.append((n, 1 if inc > 0 else 0, inc, float(imp.get(n, 0.0))))
    df = pd.DataFrame(rows, columns=["node", "has_in", "incoming", "imp"])
    if not df.empty:
        df = df.sort_values(["has_in", "incoming", "imp"], ascending=[False, False, False])
        picked = list(df["node"].head(topk))
    else:
        picked = []

    # fill if needed
    if len(picked) < topk and candidates:
        for n in candidates:
            if n in picked:
                continue
            picked.append(n)
            if len(picked) >= topk:
                break
    return picked[:topk]


def _submatrix_with_other(
    w: pd.DataFrame,
    src_keep: List[str],
    tgt_keep: List[str],
    src_other: str,
    tgt_other: str,
) -> pd.DataFrame:
    """
    Return a (len(src_keep)+1) x (len(tgt_keep)+1) matrix with OTHER row/col.
    Values are absolute weights.
    """
    w_abs = w.copy()
    try:
        w_abs = w_abs.abs()
    except Exception:
        w_abs = w_abs.applymap(lambda x: abs(_safe_float(x)))

    src_keep_set = set(src_keep)
    tgt_keep_set = set(tgt_keep)

    src_all = [s for s in w_abs.index.astype(str)]
    tgt_all = [t for t in w_abs.columns.astype(str)]

    src_other_rows = [s for s in src_all if s not in src_keep_set]
    tgt_other_cols = [t for t in tgt_all if t not in tgt_keep_set]

    # main block
    main = w_abs.reindex(index=src_keep, columns=tgt_keep, fill_value=0.0)

    # other row: sum of remaining sources -> kept targets
    other_row = pd.Series(0.0, index=tgt_keep, dtype=float)
    if len(src_other_rows) > 0:
        other_row = w_abs.reindex(index=src_other_rows, columns=tgt_keep, fill_value=0.0).sum(axis=0)

    # other col: sum of kept sources -> remaining targets
    other_col = pd.Series(0.0, index=src_keep, dtype=float)
    if len(tgt_other_cols) > 0:
        other_col = w_abs.reindex(index=src_keep, columns=tgt_other_cols, fill_value=0.0).sum(axis=1)

    # other->other: sum of remaining sources -> remaining targets
    other_other = 0.0
    if len(src_other_rows) > 0 and len(tgt_other_cols) > 0:
        other_other = float(w_abs.reindex(index=src_other_rows, columns=tgt_other_cols, fill_value=0.0).sum().sum())

    # assemble
    out = main.copy()
    out.loc[src_other, :] = other_row.values
    out.loc[:, tgt_other] = list(other_col.values) + [other_other]
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out


def _normalize_layer_flows(mat: pd.DataFrame, mode: str = "global") -> pd.DataFrame:
    """
    Normalize edge values to make Sankey readable.
    - global: scale by max within this layer
    - source: each source sums to 1 (keeps relative fan-out)
    """
    m = mat.copy().astype(float)
    m[m < 0] = 0.0
    if mode == "source":
        denom = m.sum(axis=1).replace(0.0, 1.0)
        return m.div(denom, axis=0)
    mx = float(np.nanmax(m.values)) if m.size else 1.0
    if mx <= 0:
        mx = 1.0
    return m / mx


def _prune_edges(
    mat: pd.DataFrame,
    *,
    max_edges_per_source: int = 4,
    min_value: float = 0.02,
) -> pd.DataFrame:
    """
    Keep only strongest edges per source (row-wise), plus a minimum threshold.
    Operates on a non-negative matrix (typically normalized).
    """
    if mat is None or mat.empty:
        return mat
    m = mat.copy().astype(float)
    m[m < 0] = 0.0

    keep = pd.DataFrame(0.0, index=m.index, columns=m.columns)
    for s in m.index:
        row = m.loc[s]
        row = row[row > 0]
        if row.empty:
            continue
        row = row.sort_values(ascending=False)
        row = row[row >= float(min_value)]
        if row.empty:
            continue
        if max_edges_per_source is not None and int(max_edges_per_source) > 0:
            row = row.head(int(max_edges_per_source))
        keep.loc[s, row.index] = row.values
    return keep


def _restore_edges_between_sets(
    pruned: pd.DataFrame,
    original: pd.DataFrame,
    *,
    src_nodes: List[str],
    tgt_nodes: List[str],
    exclude_prefix: Tuple[str, ...] = ("OTHER__",),
) -> pd.DataFrame:
    """
    Guarantee that edges between two displayed node sets are kept if they exist in `original`.

    This is used to satisfy the requirement:
    - nodes are strict TopK by attribution (do not change selection)
    - but if TopK(source) -> TopK(target) has a non-zero weight, we must show it (not drop due to pruning)

    Notes:
    - We do NOT invent edges; we only copy existing (non-zero) weights from `original`.
    - We ignore nodes like OTHER__* by default.
    """
    if pruned is None or pruned.empty or original is None or original.empty:
        return pruned
    p = pruned.copy()
    o = original.copy()
    p.index = p.index.astype(str)
    p.columns = p.columns.astype(str)
    o.index = o.index.astype(str)
    o.columns = o.columns.astype(str)

    src = [str(s) for s in src_nodes if not any(str(s).startswith(pref) for pref in exclude_prefix)]
    tgt = [str(t) for t in tgt_nodes if not any(str(t).startswith(pref) for pref in exclude_prefix)]

    src = [s for s in src if s in p.index and s in o.index]
    tgt = [t for t in tgt if t in p.columns and t in o.columns]
    if not src or not tgt:
        return p

    for s in src:
        for t in tgt:
            try:
                v = float(o.loc[s, t])
            except Exception:
                v = _safe_float(o.loc[s, t])
            if v > 0:
                p.loc[s, t] = max(float(p.loc[s, t]), float(v))
    return p


def _ensure_other_outgoing_to_top_targets(
    pruned: pd.DataFrame,
    original: pd.DataFrame,
    *,
    src_other: str,
    top_targets: List[str],
    min_edges: int = 1,
    max_edges: int = 3,
    exclude_targets_prefix: Tuple[str, ...] = ("OTHER__",),
) -> pd.DataFrame:
    """
    Ensure Other(hk) has at least some outgoing edges to displayed Top targets in the next layer
    (if such flow exists in the original matrix).

    We do NOT invent edges; we only restore existing edges from `original` (typically normalized).
    """
    if pruned is None or pruned.empty or original is None or original.empty:
        return pruned
    p = pruned.copy()
    o = original.copy()
    p.index = p.index.astype(str)
    p.columns = p.columns.astype(str)
    o.index = o.index.astype(str)
    o.columns = o.columns.astype(str)

    src_other = str(src_other)
    if src_other not in p.index or src_other not in o.index:
        return p

    cand = [str(t) for t in top_targets if not any(str(t).startswith(pref) for pref in exclude_targets_prefix)]
    cand = [t for t in cand if t in p.columns and t in o.columns]
    if not cand:
        return p

    # already has outgoing to any top target
    try:
        if float(p.loc[src_other, cand].sum()) > 0:
            return p
    except Exception:
        pass

    row = o.loc[src_other, cand]
    try:
        row = row.astype(float)
    except Exception:
        row = row.apply(_safe_float)
    row = row[row > 0].sort_values(ascending=False)
    if row.empty:
        return p

    k = max(int(min_edges), 1)
    if max_edges is not None and int(max_edges) > 0:
        k = min(k, int(max_edges))
    for t, v in row.head(k).items():
        p.loc[src_other, str(t)] = max(float(p.loc[src_other, str(t)]), float(v))
    return p


def _ensure_incoming_edges(
    pruned: pd.DataFrame,
    original: pd.DataFrame,
    *,
    targets: List[str],
    min_incoming: int = 1,
    exclude_targets_prefix: Tuple[str, ...] = ("OTHER__",),
) -> pd.DataFrame:
    """
    Ensure every displayed target has at least `min_incoming` incoming edges (if possible),
    otherwise Sankey will show "floating" nodes which is visually confusing and biologically misleading.
    """
    if pruned is None or pruned.empty or original is None or original.empty:
        return pruned
    p = pruned.copy()
    o = original.copy()
    o.index = o.index.astype(str)
    o.columns = o.columns.astype(str)
    p.index = p.index.astype(str)
    p.columns = p.columns.astype(str)

    for t in [str(x) for x in targets]:
        if any(t.startswith(pref) for pref in exclude_targets_prefix):
            continue
        if t not in o.columns or t not in p.columns:
            continue
        # already has incoming?
        if float(p[t].sum()) > 0:
            continue
        col = o[t]
        col = col[col > 0]
        if col.empty:
            continue
        # add top incoming(s)
        for s, v in col.sort_values(ascending=False).head(int(min_incoming)).items():
            if s in p.index:
                p.loc[s, t] = max(float(p.loc[s, t]), float(v))
    return p


def _initial_layer_order(layer_nodes: List[str], importance: Optional[pd.Series] = None) -> List[str]:
    """
    Start with importance order (desc), keep OTHER__* at the bottom.
    """
    nodes = [str(n) for n in layer_nodes]
    other = [n for n in nodes if n.startswith("OTHER__")]
    main = [n for n in nodes if not n.startswith("OTHER__")]
    if importance is not None and not importance.empty:
        imp = importance.copy()
        imp.index = imp.index.astype(str)
        main = sorted(main, key=lambda x: float(imp.get(x, 0.0)), reverse=True)
    return main + other


def _barycenter_reorder(
    src_order: List[str],
    tgt_order: List[str],
    mat_src_tgt: pd.DataFrame,
    *,
    reorder_target: bool,
) -> List[str]:
    """
    One sweep of barycenter ordering to reduce crossings.
    """
    if mat_src_tgt is None or mat_src_tgt.empty:
        return tgt_order if reorder_target else src_order

    src_pos = {n: i for i, n in enumerate(src_order)}
    tgt_pos = {n: i for i, n in enumerate(tgt_order)}

    def bary_for_target(t: str) -> float:
        col = mat_src_tgt.get(t)
        if col is None:
            return float(tgt_pos.get(t, 0))
        col = col[col > 0]
        if col.empty:
            return float(tgt_pos.get(t, 0))
        num = 0.0
        den = 0.0
        for s, w in col.items():
            num += float(w) * float(src_pos.get(str(s), 0))
            den += float(w)
        if den <= 0:
            return float(tgt_pos.get(t, 0))
        return num / den

    def bary_for_source(s: str) -> float:
        row = mat_src_tgt.loc[s] if s in mat_src_tgt.index else None
        if row is None is True:
            return float(src_pos.get(s, 0))
        row = row[row > 0]
        if row.empty:
            return float(src_pos.get(s, 0))
        num = 0.0
        den = 0.0
        for t, w in row.items():
            num += float(w) * float(tgt_pos.get(str(t), 0))
            den += float(w)
        if den <= 0:
            return float(src_pos.get(s, 0))
        return num / den

    if reorder_target:
        other = [n for n in tgt_order if n.startswith("OTHER__")]
        main = [n for n in tgt_order if not n.startswith("OTHER__")]
        main = sorted(main, key=bary_for_target)
        return main + other
    else:
        other = [n for n in src_order if n.startswith("OTHER__")]
        main = [n for n in src_order if not n.startswith("OTHER__")]
        main = sorted(main, key=bary_for_source)
        return main + other


def _layer_y_positions(layer_order: List[str], node_size: Dict[str, float], *, gap: float = 0.12) -> Dict[str, float]:
    """
    Allocate y positions within a layer using node_size as vertical weight.
    """
    sizes = [max(0.2, float(node_size.get(n, 0.2))) for n in layer_order]
    total = float(sum(sizes) + gap * (len(layer_order) - 1)) if layer_order else 1.0
    y_map: Dict[str, float] = {}
    acc = 0.0
    for n, sz in zip(layer_order, sizes):
        y_map[n] = (acc + sz / 2.0) / total
        acc += sz + gap
    return y_map


def run(
    output_dir: str,
    node_weights: Dict[str, pd.DataFrame],
    link_weights: Dict[str, pd.DataFrame],
    h_keys: List[str],
    *,
    topk_per_layer: int = 10,
    flow_normalization: str = "source",
    max_edges_per_source: int = 4,
    min_edge_value: float = 0.03,
    ordering_iterations: int = 0,
    ordering_mode: str = "rank",
    ensure_connectivity: bool = True,
    connectivity_topk: int = 6,
    include_input_layer: bool = False,
    include_input_other: bool = False,
    candidate_pool: int = 50,
    title: str = "Hybrid network Sankey (KG-enhanced)",
) -> str:
    """
    Build a tidy Sankey HTML and save it under output_dir.

    Args:
        output_dir: Config output dir (results/...).
        node_weights: OrderedDict like {"inputs": df, "h0": df, ...} with 'coef'.
        link_weights: Dict like {"h0_to_h1": df, ..., "hK_to_root": df}.
        h_keys: ordered ["h0","h1",...].

    Returns:
        Path to the generated HTML.
    """
    extracted_dir = join(output_dir, "extracted")
    _ensure_dir(extracted_dir)

    reactome_id_to_name = _read_reactome_id_to_name()
    reactome_short_name = _read_pathway_short_names()
    string_id_to_name = _read_string_id_to_preferred_name()

    # Nodes per layer: strict TopK by attribution (plus one Other bucket).
    # We do NOT change TopK based on connectivity. Connectivity is handled at the edge-rendering stage
    # by guaranteeing that TopK->TopK edges are not removed by pruning if they exist.
    layer_nodes = _build_layer_nodes(h_keys=h_keys, node_weights=node_weights, topk=int(topk_per_layer))

    # Build per-layer matrices first, so we can (a) prune edges and (b) reorder nodes to reduce crossings.
    mats: List[pd.DataFrame] = []
    for i, hk in enumerate(h_keys):
        src_nodes = layer_nodes[hk]
        src_other = f"OTHER__{hk}"

        if i < len(h_keys) - 1:
            hk2 = h_keys[i + 1]
            tgt_nodes = layer_nodes[hk2]
            tgt_other = f"OTHER__{hk2}"
            key = f"{hk}_to_{hk2}"
        else:
            tgt_nodes = ["root"]
            tgt_other = "root"
            key = f"{hk}_to_root"

        if key not in link_weights:
            mats.append(pd.DataFrame(0.0, index=src_nodes, columns=tgt_nodes))
            continue
        w = link_weights[key].copy()
        w.index = w.index.astype(str)
        w.columns = w.columns.astype(str)
        mat = _submatrix_with_other(
            w=w,
            src_keep=[n for n in src_nodes if n != src_other],
            tgt_keep=[n for n in tgt_nodes if n != tgt_other],
            src_other=src_other,
            tgt_other=tgt_other,
        )
        mat = _normalize_layer_flows(mat, mode=flow_normalization)
        mat_orig = mat.copy()
        mat = _prune_edges(mat, max_edges_per_source=max_edges_per_source, min_value=min_edge_value)
        # Key rule: if TopK(source)->TopK(target) has a non-zero connection, it must be shown explicitly.
        mat = _restore_edges_between_sets(
            pruned=mat,
            original=mat_orig,
            src_nodes=[n for n in src_nodes if not str(n).startswith("OTHER__")],
            tgt_nodes=[n for n in tgt_nodes if (n == "root") or (not str(n).startswith("OTHER__"))],
        )
        # Ensure Other(hk) shows some flow to next-layer Top targets if such flow exists
        # (otherwise pruning may leave it only connected to Other(h{k+1})).
        mat = _ensure_other_outgoing_to_top_targets(
            pruned=mat,
            original=mat_orig,
            src_other=src_other,
            top_targets=[n for n in tgt_nodes if (n == "root") or (not str(n).startswith("OTHER__"))],
            min_edges=1,
            max_edges=3,
        )
        # Critical: if we show a target node (e.g., top proteins), it must have at least one incoming edge.
        mat = _ensure_incoming_edges(
            pruned=mat,
            original=mat_orig,
            targets=[n for n in tgt_nodes if not str(n).startswith("OTHER__")],
            min_incoming=1,
        )
        mats.append(mat)

    # Initial ordering by importance within each layer (OTHER at bottom)
    orders: List[List[str]] = []
    for i, hk in enumerate(h_keys):
        imp = _node_importance_series(node_weights, hk)
        orders.append(_initial_layer_order(layer_nodes[hk], imp))
    orders.append(["root"])

    # Optional: prepend an input-feature layer (mut/cnv_amp/cnv_del) using gradient_importance_0.csv.
    # This creates: inputs -> h0 (genes) with an "Other (inputs)" bucket.
    # It is off by default to keep the graph compact.
    input_links_df: Optional[pd.DataFrame] = None
    input_nodes: Optional[List[str]] = None
    if include_input_layer:
        try:
            gi0_path = join(extracted_dir, "gradient_importance_0.csv")
            gi0 = pd.read_csv(gi0_path, index_col=[0, 1])
            if "coef" in gi0.columns:
                s = gi0["coef"].abs()
            else:
                s = gi0.iloc[:, 0].abs()
            # index is (gene, feature_type)
            s = s.reset_index()
            s.columns = ["gene", "feature", "coef"]

            # Aggregate by feature type and gene
            piv = s.pivot_table(index="feature", columns="gene", values="coef", aggfunc="sum").fillna(0.0)
            # Keep the most common 3 features; put others into Other
            feature_order = ["mut_important", "cnv_amp", "cnv_del"]
            present = [f for f in feature_order if f in piv.index]
            other_feats = [f for f in piv.index if f not in set(present)]

            # Friendly display names (match your expectation)
            rename_feat = {"mut_important": "mutation", "cnv_amp": "amplification", "cnv_del": "deletion"}
            present_disp = [rename_feat.get(f, f) for f in present]
            input_nodes = present_disp + (["Other (inputs)"] if include_input_other else [])
            # Build links: feature -> gene, normalize per gene
            links = []
            hk0 = h_keys[0]
            other_h0 = f"OTHER__{hk0}"
            gene_top = [n for n in layer_nodes[hk0] if not str(n).startswith("OTHER__")]
            gene_top_set = set(map(str, gene_top))
            other_gene_list = [str(g) for g in piv.columns.astype(str) if str(g) not in gene_top_set]

            for g in piv.columns:
                col = piv[g]
                denom = float(col.sum()) if float(col.sum()) > 0 else 1.0
                # Only link to explicitly shown TopK genes; the rest are aggregated into Other(h0).
                if str(g) in gene_top_set:
                    for f, disp in zip(present, present_disp):
                        v = float(col.get(f, 0.0)) / denom
                        if v <= 0:
                            continue
                        links.append({"source": disp, "target": str(g), "value": v})
                    # other bucket
                    if include_input_other and other_feats:
                        v = float(col.reindex(other_feats).sum()) / denom
                        if v > 0:
                            links.append({"source": "Other (inputs)", "target": str(g), "value": v})

            # Also connect feature layer -> Other(h0) to represent all non-TopK genes.
            # This satisfies: gene-layer Other must have incoming from feature layer.
            if other_gene_list and present:
                denom_other = float(piv.reindex(index=present, columns=other_gene_list).sum().sum())
                if denom_other <= 0:
                    denom_other = 1.0
                for f, disp in zip(present, present_disp):
                    v = float(piv.reindex(index=[f], columns=other_gene_list).sum().sum()) / denom_other
                    if v > 0:
                        links.append({"source": disp, "target": other_h0, "value": v})
                if include_input_other and other_feats:
                    v = float(piv.reindex(index=other_feats, columns=other_gene_list).sum().sum()) / denom_other
                    if v > 0:
                        links.append({"source": "Other (inputs)", "target": other_h0, "value": v})

            input_links_df = pd.DataFrame(links)
        except Exception:
            input_links_df = None
            input_nodes = None

    # Ordering:
    # - "rank" (default): keep strict attribution ranking top->bottom (Other at bottom).
    # - "barycenter": reorder to reduce crossings (may change visual order vs ranking).
    if str(ordering_mode).lower() == "barycenter":
        iters = max(0, int(ordering_iterations))
        for _ in range(iters):
            # forward: reorder targets based on incoming from previous
            for li in range(len(h_keys) - 1):
                mat = mats[li]
                orders[li + 1] = _barycenter_reorder(orders[li], orders[li + 1], mat, reorder_target=True)
            # backward: reorder sources based on outgoing to next
            for li in range(len(h_keys) - 2, -1, -1):
                mat = mats[li]
                orders[li] = _barycenter_reorder(orders[li], orders[li + 1], mat, reorder_target=False)

    # Node sizes (for y spacing): use max(in_sum, out_sum) over pruned mats
    node_size: Dict[str, float] = {}
    for li, mat in enumerate(mats):
        if mat is None or mat.empty:
            continue
        out_sum = mat.sum(axis=1).to_dict()
        in_sum = mat.sum(axis=0).to_dict()
        for n, v in out_sum.items():
            node_size[str(n)] = max(float(node_size.get(str(n), 0.0)), float(v))
        for n, v in in_sum.items():
            node_size[str(n)] = max(float(node_size.get(str(n), 0.0)), float(v))
    node_size["root"] = max(float(node_size.get("root", 0.0)), 1.0)

    # build global node list with stable ordering by layer then within-layer order
    node_ids: List[str] = []
    node_layer_index: Dict[str, int] = {}
    for li, layer in enumerate(orders):
        for nid in layer:
            if nid not in node_layer_index:
                node_layer_index[nid] = li
                node_ids.append(nid)

    # positions by (layer, within-layer y map)
    n_layers = len(orders)  # includes root layer
    x_map: Dict[str, float] = {}
    y_map: Dict[str, float] = {}
    for li, layer in enumerate(orders):
        x = 0.0 if n_layers == 1 else li / (n_layers - 1)
        for nid in layer:
            x_map[nid] = x
        ym = _layer_y_positions(layer, node_size, gap=0.10)
        y_map.update(ym)

    xs = [float(x_map.get(n, 0.0)) for n in node_ids]
    ys = [float(y_map.get(n, 0.5)) for n in node_ids]

    rank_map = _rank_map_for_layers(h_keys=h_keys, node_weights=node_weights, topk=int(topk_per_layer))
    labels = [
        _label_with_rank(
            _label_for_node(nid, reactome_id_to_name, reactome_short_name, string_id_to_name),
            nid,
            rank_map,
        )
        for nid in node_ids
    ]
    types = [
        _infer_node_type(nid.replace("OTHER__", "")) if nid.startswith("OTHER__") else _infer_node_type(nid)
        for nid in node_ids
    ]
    node_colors = [_node_color(t) for t in types]
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    # emit edges from pruned matrices
    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []
    edge_colors: List[str] = []

    # optional inputs -> genes
    if include_input_layer and input_links_df is not None and input_nodes is not None:
        # Make sure input nodes are inserted as a new leftmost layer in the Plotly diagram:
        # Plotly's Sankey doesn't let us add a new layer after we already built node_ids;
        # so we encode them as normal nodes here if missing.
        for n in input_nodes:
            if n not in node_index:
                node_layer_index[n] = -1  # inputs
        # If we inserted new nodes, rebuild node_ids and node_index in a stable way.
        if any(li == -1 for li in node_layer_index.values()):
            # prepend input layer order
            input_order = input_nodes
            orders.insert(0, input_order)
            # rebuild node list
            node_ids = []
            node_layer_index = {}
            for li, layer in enumerate(orders):
                for nid in layer:
                    if nid not in node_layer_index:
                        node_layer_index[nid] = li
                        node_ids.append(nid)
            n_layers = len(orders)
            x_map = {}
            y_map = {}
            for li, layer in enumerate(orders):
                x = 0.0 if n_layers == 1 else li / (n_layers - 1)
                for nid in layer:
                    x_map[nid] = x
                ym = _layer_y_positions(layer, node_size, gap=0.10)
                y_map.update(ym)
            xs = [float(x_map.get(n, 0.0)) for n in node_ids]
            ys = [float(y_map.get(n, 0.5)) for n in node_ids]
            rank_map = _rank_map_for_layers(h_keys=h_keys, node_weights=node_weights, topk=int(topk_per_layer))
            labels = [
                _label_with_rank(
                    _label_for_node(nid, reactome_id_to_name, reactome_short_name, string_id_to_name),
                    nid,
                    rank_map,
                )
                for nid in node_ids
            ]
            types = [
                _infer_node_type(nid.replace("OTHER__", "")) if str(nid).startswith("OTHER__") else _infer_node_type(str(nid))
                for nid in node_ids
            ]
            # force input feature nodes to a distinct color
            node_colors = []
            for nid, t in zip(node_ids, types):
                if str(nid) in ("mutation", "amplification", "deletion", "Other (inputs)"):
                    node_colors.append("rgba(105,189,210,0.75)")
                else:
                    node_colors.append(_node_color(t))
            node_index = {nid: i for i, nid in enumerate(node_ids)}

        # add links
        for _, r in input_links_df.iterrows():
            s = str(r["source"])
            t = str(r["target"])
            v = float(r["value"])
            if v <= 0:
                continue
            sid = node_index.get(s)
            tid = node_index.get(t)
            if sid is None or tid is None:
                continue
            sources.append(sid)
            targets.append(tid)
            values.append(v)
            edge_colors.append(_edge_color("rgba(105,189,210,0.75)", alpha=0.18))

    for li, hk in enumerate(h_keys):
        mat = mats[li]
        if mat is None or mat.empty:
            continue
        for s in mat.index:
            for t in mat.columns:
                v = float(mat.loc[s, t])
                if v <= 0:
                    continue
                sid = node_index.get(str(s))
                tid = node_index.get(str(t))
                if sid is None or tid is None:
                    continue
                sources.append(sid)
                targets.append(tid)
                values.append(v)
                edge_colors.append(_edge_color(node_colors[sid], alpha=0.22))

    # export debug tables (helps validating “top10 对不上”问题)
    nodes_debug = pd.DataFrame(
        {
            "node_id": node_ids,
            "label": labels,
            "layer": [node_layer_index[n] for n in node_ids],
            "type": types,
            "x": xs,
            "y": ys,
        }
    )
    nodes_debug.to_csv(join(extracted_dir, "sankey_hybrid_nodes.csv"), index=False)
    links_debug = pd.DataFrame(
        {
            "source": [node_ids[i] for i in sources],
            "target": [node_ids[i] for i in targets],
            "value": values,
        }
    )
    links_debug.to_csv(join(extracted_dir, "sankey_hybrid_links.csv"), index=False)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(60,60,60,0.35)", width=0.5),
                    label=labels,
                    color=node_colors,
                    x=xs,
                    y=ys,
                ),
                link=dict(source=sources, target=targets, value=values, color=edge_colors),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        font=dict(size=12),
        margin=dict(l=20, r=20, t=55, b=20),
        height=max(520, 160 * n_layers),
        width=1100,
    )

    html_path = join(output_dir, "sankey_hybrid.html")
    fig.write_html(html_path, include_plotlyjs="cdn")

    # Try static images if kaleido is available; keep best-effort.
    try:
        fig.write_image(join(output_dir, "sankey_hybrid.png"), scale=2)
    except Exception:
        pass
    try:
        fig.write_image(join(output_dir, "sankey_hybrid.pdf"))
    except Exception:
        pass

    return html_path


def run_from_extracted(
    output_dir: str,
    *,
    topk_per_layer: int = 10,
    flow_normalization: str = "source",
    max_edges_per_source: int = 4,
    min_edge_value: float = 0.03,
    ordering_iterations: int = 0,
    ordering_mode: str = "rank",
    ensure_connectivity: bool = True,
    connectivity_topk: int = 6,
    include_input_layer: bool = False,
    include_input_other: bool = False,
    candidate_pool: int = 50,
    title: str = "Hybrid network Sankey (KG-enhanced)",
) -> str:
    """
    Re-render Sankey from already-exported files in `${output_dir}/extracted/`.

    Expected files (as produced by `interpret_propathnet_model.py`):
    - extracted/node_importance_graph_adjusted.csv
    - extracted/link_weights_1.csv ... extracted/link_weights_K.csv

    This is useful when a run exported importance/link CSVs but skipped Sankey rendering
    (e.g., config flag defaulting to None).
    """
    extracted_dir = join(output_dir, "extracted")
    node_imp_path = join(extracted_dir, "node_importance_graph_adjusted.csv")
    if not exists(node_imp_path):
        raise FileNotFoundError(f"Missing {node_imp_path}")

    node_imp = pd.read_csv(node_imp_path, index_col=0)
    if "layer" not in node_imp.columns:
        raise ValueError("node_importance_graph_adjusted.csv missing 'layer' column")
    if "coef" not in node_imp.columns:
        # allow legacy naming
        if "coef_combined" in node_imp.columns:
            node_imp["coef"] = node_imp["coef_combined"]
        else:
            raise ValueError("node_importance_graph_adjusted.csv missing 'coef' column")

    # determine how many link files exist
    link_nums: List[int] = []
    for fn in os.listdir(extracted_dir):
        m = re.match(r"link_weights_(\d+)\.csv$", fn)
        if m:
            link_nums.append(int(m.group(1)))
    link_nums = sorted(link_nums)
    if not link_nums:
        raise FileNotFoundError(f"No link_weights_*.csv found in {extracted_dir}")

    n_links = max(link_nums)
    # With our export convention: link_weights_1..link_weights_{N-1} connect h0->h1..h{N-2}->h{N-1},
    # and link_weights_N connects h{N-1}->root.
    h_keys = [f"h{i}" for i in range(n_links)]

    # reconstruct node_weights per h*
    node_weights: Dict[str, pd.DataFrame] = {}
    for i, hk in enumerate(h_keys):
        layer_num = i + 1  # h0 => 1
        df = node_imp[node_imp["layer"] == layer_num].copy()
        if df.empty:
            node_weights[hk] = pd.DataFrame(columns=["coef"])
            continue
        s = df["coef"].apply(_safe_float).abs()
        node_weights[hk] = pd.DataFrame({"coef": s.values}, index=df.index.astype(str)).sort_values("coef", ascending=False)

    # reconstruct link_weights dict in key form expected by `run`
    link_weights: Dict[str, pd.DataFrame] = {}
    for i in range(1, n_links + 1):
        p = join(extracted_dir, f"link_weights_{i}.csv")
        w = pd.read_csv(p, index_col=0)
        w.index = w.index.astype(str)
        w.columns = w.columns.astype(str)
        if i < n_links:
            link_weights[f"h{i-1}_to_h{i}"] = w
        else:
            link_weights[f"h{i-1}_to_root"] = w

    return run(
        output_dir=output_dir,
        node_weights=node_weights,
        link_weights=link_weights,
        h_keys=h_keys,
        topk_per_layer=topk_per_layer,
        flow_normalization=flow_normalization,
        max_edges_per_source=max_edges_per_source,
        min_edge_value=min_edge_value,
        ordering_iterations=ordering_iterations,
        ordering_mode=ordering_mode,
        ensure_connectivity=ensure_connectivity,
        connectivity_topk=connectivity_topk,
        include_input_layer=include_input_layer,
        include_input_other=include_input_other,
        candidate_pool=candidate_pool,
        title=title,
    )

