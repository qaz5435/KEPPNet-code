"""
Export the hybrid Sankey (from extracted CSVs) to an ECharts option JSON.

Input (produced by `sankey_hybrid.py`):
  - {output_dir}/extracted/sankey_hybrid_nodes.csv
  - {output_dir}/extracted/sankey_hybrid_links.csv

Output:
  - {output_dir}/sankey_hybrid_echarts.json
"""

from __future__ import annotations

import json
import re
import sys
from os.path import join, exists
from typing import Dict, List

import pandas as pd


def _node_display_name(node_id: str, label: str) -> str:
    # ECharts Sankey identifies nodes by name; must be unique.
    if str(node_id).startswith("OTHER__"):
        # OTHER__h3 -> Other (h3)
        suffix = str(node_id).split("__", 1)[-1]
        return f"Other ({suffix})"
    if str(node_id) == "root":
        return "Outcome"
    # usually unique: genes or pathway names
    return str(label) if str(label).strip() else str(node_id)


def _rgba_to_echarts(color: str) -> str:
    """
    Accepts 'rgba(r,g,b,a)' or any CSS color string.
    """
    return str(color)


def _level_style(depth: int) -> Dict:
    # Simple alternating style; genes are usually depth 0 in our export.
    palette = [
        "rgba(60,140,220,0.75)",   # gene-ish
        "rgba(120,90,200,0.75)",   # pathway-ish
        "rgba(240,140,60,0.75)",   # protein-ish (if present)
        "rgba(120,90,200,0.75)",
        "rgba(60,140,220,0.75)",
        "rgba(120,90,200,0.75)",
    ]
    c = palette[depth % len(palette)]
    return {
        "depth": int(depth),
        "itemStyle": {"color": c},
        "lineStyle": {"color": "source", "opacity": 0.18, "curveness": 0.5},
        "label": {"color": "#2b2b2b", "fontSize": 11},
    }


def export(output_dir: str) -> str:
    nodes_path = join(output_dir, "extracted", "sankey_hybrid_nodes.csv")
    links_path = join(output_dir, "extracted", "sankey_hybrid_links.csv")
    if not exists(nodes_path):
        raise FileNotFoundError(nodes_path)
    if not exists(links_path):
        raise FileNotFoundError(links_path)

    nodes_df = pd.read_csv(nodes_path)
    links_df = pd.read_csv(links_path)

    # Build unique names for ECharts
    id_to_name: Dict[str, str] = {}
    name_to_depth: Dict[str, int] = {}
    name_to_color: Dict[str, str] = {}

    for _, r in nodes_df.iterrows():
        node_id = str(r.get("node_id"))
        label = str(r.get("label", node_id))
        depth = int(r.get("layer", 0))
        color = str(r.get("color", "")) if "color" in nodes_df.columns else ""

        name = _node_display_name(node_id, label)
        # guarantee uniqueness: if duplicated, suffix with node_id
        if name in name_to_depth:
            name = f"{name} [{node_id}]"

        id_to_name[node_id] = name
        name_to_depth[name] = depth
        if color:
            name_to_color[name] = _rgba_to_echarts(color)

    data: List[Dict] = []
    for node_id, name in id_to_name.items():
        depth = int(name_to_depth.get(name, 0))
        item = {"name": name, "depth": depth}
        if name in name_to_color:
            item["itemStyle"] = {"color": name_to_color[name]}
        data.append(item)

    # Links
    links: List[Dict] = []
    for _, r in links_df.iterrows():
        src_id = str(r.get("source"))
        tgt_id = str(r.get("target"))
        v = float(r.get("value", 0.0))
        if v <= 0:
            continue
        src = id_to_name.get(src_id, src_id)
        tgt = id_to_name.get(tgt_id, tgt_id)
        links.append({"source": src, "target": tgt, "value": v})

    max_depth = int(nodes_df["layer"].max()) if "layer" in nodes_df.columns and not nodes_df.empty else 0
    levels = [_level_style(d) for d in range(max_depth + 1)]

    option = {
        "backgroundColor": "#ffffff",
        "title": {"text": "Hybrid network Sankey (KG-enhanced)", "left": "left"},
        "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
        "series": [
            {
                "type": "sankey",
                # Give labels and nodes more room to avoid overlaps.
                "left": "2%",
                "right": "18%",
                "top": "8%",
                "bottom": "6%",
                "nodeWidth": 12,
                "nodeGap": 18,
                "nodeAlign": "justify",
                # Default: keep layout stable (no auto-iterations) but allow manual dragging.
                "draggable": True,
                # Keep a stable layout; set >0 if you want auto packing.
                "layoutIterations": 0,
                "emphasis": {"focus": "adjacency"},
                "data": data,
                "links": links,
                "levels": levels,
                "lineStyle": {"color": "source", "opacity": 0.18, "curveness": 0.5},
                # Truncate very long pathway names; you can show full text via tooltip.
                "label": {"color": "#2b2b2b", "fontSize": 11, "overflow": "truncate", "width": 220},
            }
        ],
    }

    out_path = join(output_dir, "sankey_hybrid_echarts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(option, f, ensure_ascii=False, indent=2)
    return out_path


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m src.utils.export_sankey_echarts <output_dir>")
        return 2
    output_dir = argv[1]
    out_path = export(output_dir)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

