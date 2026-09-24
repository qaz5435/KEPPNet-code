"""
Attribution stability experiment (ProPathNet / reactome_protein_kg).

Goal
----
Run N independent trainings (different random seeds), and after each training:
- read attribution outputs from `${output_dir}/extracted/`
- extract Top-10 nodes for the first three biology layers starting from gene:
    Gene layer  -> Pathway layer -> Protein layer
- save all runs' Top-10 lists and stability summaries.

Why a separate script?
----------------------
Earlier stability scripts edited a shared configuration file in place to change seeds.
This script avoids mutating global config files by using Config(config_dict=...) overrides per run.

Outputs
-------
All outputs are written under:
  results/attribution_stability/<exp_id>/

Files:
  - manifest.json
  - attribution_top10_long.csv            (tidy per-run, per-layer, per-rank table)
  - stability_jaccard.csv                 (mean/std of pairwise Jaccard for each layer)
  - node_frequency.csv                    (node frequency + avg rank per layer)

Usage
-----
python -m src.interpretability.summarize_multi_attribution \\
  --model prad-main \\
  --dataset prostate_kg \\
  --n-runs 10 \\
  --seed-base 2024 \\
  --seed-step 100 \\
  --prepare-data \\
  --study-names prad_p1000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from os.path import join, exists
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.configuration import Config
from src.data.dataload import BnetDataLoader
from src.data.dataset import BnetData, set_cached_data_to_empty
from src.models.pathpronet import Model
from src.training.trainer import Trainer
from src.utils.logger import init_logger
from src.utils.utils import init_seed

# interpretability dispatch must match run_me.py behavior
from src.interpretability.attribution_methods import interpret as interpret_model_copy
from src.interpretability.attribution_methods import interpret_propathnet_model


_RE_REACTOME = re.compile(r"^R-HSA-\d+$")
_RE_STRING_PROTEIN = re.compile(r"^9606\.ENSP\d+$")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _infer_bio_type(node_id: str) -> str:
    """
    Infer node type from node_id patterns.
    Returns: "gene" | "pathway" | "protein" | "other"
    """
    s = str(node_id)
    if _RE_REACTOME.match(s):
        return "pathway"
    if _RE_STRING_PROTEIN.match(s):
        return "protein"
    # gene symbol-like fallback
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{0,30}$", s):
        return "gene"
    return "other"


def _read_topk_csv(path: str) -> pd.DataFrame:
    """
    Read `${output_dir}/extracted/topk_layer_{k}.csv` exported by interpret_propathnet_model.
    Expected columns: score, rank (index = node_id).
    """
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    # normalize columns
    if "score" not in df.columns and "coef" in df.columns:
        df = df.rename(columns={"coef": "score"})
    if "rank" not in df.columns:
        df["rank"] = range(1, len(df) + 1)
    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)
    # enforce rank order
    df = df.sort_values("rank", ascending=True)
    return df


def _layer_type_from_topk(df: pd.DataFrame) -> str:
    """
    Determine layer type by majority vote among its TopK node IDs.
    """
    if df is None or df.empty:
        return "other"
    types = [_infer_bio_type(x) for x in df.index.astype(str)]
    c = pd.Series(types).value_counts()
    if c.empty:
        return "other"
    # tie-breaker preference: gene > pathway > protein > other
    pref = {"gene": 3, "pathway": 2, "protein": 1, "other": 0}
    best = sorted(c.index.tolist(), key=lambda t: (int(c[t]), pref.get(t, 0)), reverse=True)[0]
    return str(best)


def _extract_gene_pathway_protein_topk(output_dir: str, *, topk: int = 10) -> Dict[str, Dict]:
    """
    From one run's output_dir, pick three layers starting from the first gene layer:
      gene -> pathway -> protein
    Returns:
      {
        "gene":   {"layer_num": int, "df": DataFrame},
        "pathway":{"layer_num": int, "df": DataFrame},
        "protein":{"layer_num": int, "df": DataFrame},
      }
    """
    extracted_dir = join(os.getcwd(), output_dir, "extracted")
    if not exists(extracted_dir):
        raise FileNotFoundError(f"Missing extracted dir: {extracted_dir}")

    # collect all topk_layer_*.csv
    layer_files: List[Tuple[int, str]] = []
    for fn in os.listdir(extracted_dir):
        m = re.match(r"topk_layer_(\d+)\.csv$", fn)
        if m:
            layer_files.append((int(m.group(1)), join(extracted_dir, fn)))
    layer_files.sort(key=lambda x: x[0])
    if not layer_files:
        raise FileNotFoundError(f"No topk_layer_*.csv in {extracted_dir}")

    layers: List[Tuple[int, str, pd.DataFrame]] = []
    for layer_num, path in layer_files:
        df = _read_topk_csv(path)
        df = df.head(int(topk)).copy()
        layers.append((layer_num, _layer_type_from_topk(df), df))

    # find first gene layer, then next pathway, then next protein (in order)
    start_idx = None
    for i, (_, t, _) in enumerate(layers):
        if t == "gene":
            start_idx = i
            break
    if start_idx is None:
        raise RuntimeError(f"Could not find a gene layer in topk layers under {output_dir}")

    def _find_next(target_type: str, from_i: int) -> Optional[int]:
        for j in range(from_i + 1, len(layers)):
            if layers[j][1] == target_type:
                return j
        return None

    idx_gene = start_idx
    idx_pathway = _find_next("pathway", idx_gene)
    idx_protein = _find_next("protein", idx_pathway if idx_pathway is not None else idx_gene)

    if idx_pathway is None or idx_protein is None:
        # Provide a helpful debugging hint (layer types observed).
        observed = [{"layer_num": n, "type": t} for n, t, _ in layers]
        raise RuntimeError(
            "Failed to locate gene->pathway->protein layers. "
            f"Observed layer types: {observed}. "
            "If your architecture differs, adjust the selection logic accordingly."
        )

    out: Dict[str, Dict] = {}
    for key, idx in [("gene", idx_gene), ("pathway", idx_pathway), ("protein", idx_protein)]:
        layer_num, _, df = layers[idx]
        out[key] = {"layer_num": int(layer_num), "df": df.copy()}
    return out


def _pairwise_jaccard(list_of_sets: List[set]) -> List[float]:
    vals: List[float] = []
    for i in range(len(list_of_sets)):
        for j in range(i + 1, len(list_of_sets)):
            a, b = list_of_sets[i], list_of_sets[j]
            denom = len(a | b)
            vals.append((len(a & b) / denom) if denom else 0.0)
    return vals


@dataclass
class RunRecord:
    run_id: int
    seed: int
    output_dir: str
    status: str  # "ok" | "failed"
    error: Optional[str] = None


def _run_one_training(*, model_name: str, dataset_name: str, seed: int, output_dir: str) -> Dict[str, object]:
    """
    Train + test + interpretability once, with per-run overrides.
    Returns the `trainer.test()` result dict.
    """
    # Make sure output dir exists early (logger may want it).
    _ensure_dir(join(os.getcwd(), output_dir))

    # Override config without mutating global json files.
    overrides = {
        "random_seed": int(seed),
        "output_dir": str(output_dir),  # Trainer joins with cwd internally
        # Keep logs next to outputs for each run.
        "log_file": join(str(output_dir), "run.log"),
    }
    config = Config(model_name, dataset_name, config_dict=overrides)

    init_seed(config["random_seed"], config["reproducibility"])
    init_logger(config)

    dataset = BnetData(config)
    dataloader = BnetDataLoader(config, dataset)
    model = Model(config, dataset)

    trainer = Trainer(config, dataloader, model)
    trainer.fix()
    result = trainer.test()

    # interpretability
    if str(config["network"]).lower() == "reactome_protein_kg":
        interpret_propathnet_model(config, model, dataloader)
    else:
        interpret_model_copy(config, model, dataloader)

    # cleanup (important when running multiple times in one process)
    del config, dataset, dataloader, model, trainer
    set_cached_data_to_empty()

    return result


def run_experiment(
    *,
    model_name: str,
    dataset_name: str,
    n_runs: int = 10,
    seed_base: int = 2024,
    seed_step: int = 100,
    topk: int = 10,
    exp_root: str = "results/attribution_stability",
    exp_id: Optional[str] = None,
    prepare_data_flag: bool = False,
    study_names: Optional[List[str]] = None,
) -> str:
    """
    Returns the experiment directory path (relative to repo root).
    """
    if exp_id is None:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name))
        safe_data = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name))
        exp_id = f"{ts}-{safe_data}-{safe_model}-attr-stability"

    exp_dir = join(exp_root, exp_id)
    runs_dir = join(exp_dir, "runs")
    _ensure_dir(join(os.getcwd(), runs_dir))

    if prepare_data_flag:
        from src.data.processor.medical_data.prepare_data import prepare_data
        from src.data.processor.medical_data.split_data import split_data

        if not study_names:
            raise ValueError("--prepare-data requires --study-names")
        cancer_type = "breast" if "breast" in dataset_name.lower() else "prostate"
        cohort = "brca" if cancer_type == "breast" else "prad"
        prepare_data(study_names, cancer_type=cancer_type)
        split_data(
            processed_dir=join("data", "structured", cohort, "processed"),
            output_dir=join("data", "structured", cohort, "splits"),
        )

    records: List[RunRecord] = []
    rows_long: List[Dict[str, object]] = []

    for i in range(int(n_runs)):
        run_id = i + 1
        seed = int(seed_base) + i * int(seed_step)
        out_dir = join(runs_dir, f"run_{run_id:02d}_seed_{seed}")

        rec = RunRecord(run_id=run_id, seed=seed, output_dir=out_dir, status="failed", error=None)
        try:
            _run_one_training(model_name=model_name, dataset_name=dataset_name, seed=seed, output_dir=out_dir)
            layers = _extract_gene_pathway_protein_topk(out_dir, topk=topk)

            for layer_type in ["gene", "pathway", "protein"]:
                layer_num = int(layers[layer_type]["layer_num"])
                df = layers[layer_type]["df"]
                for node_id, r in df.iterrows():
                    rows_long.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "output_dir": out_dir,
                            "layer_type": layer_type,
                            "layer_num": layer_num,
                            "rank": int(r["rank"]),
                            "node": str(node_id),
                            "score": float(r["score"]),
                        }
                    )

            rec.status = "ok"
        except Exception as e:
            rec.error = repr(e)
        records.append(rec)

    # Save manifest + per-run topk
    manifest = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_runs": int(n_runs),
        "seed_base": int(seed_base),
        "seed_step": int(seed_step),
        "topk": int(topk),
        "exp_dir": exp_dir,
        "runs": [asdict(r) for r in records],
    }
    with open(join(os.getcwd(), exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    topk_long = pd.DataFrame(rows_long)
    topk_long.to_csv(join(os.getcwd(), exp_dir, "attribution_top10_long.csv"), index=False)

    # Stability summaries (only on successful runs)
    ok_runs = topk_long["run_id"].unique().tolist() if not topk_long.empty else []
    if ok_runs:
        st_rows: List[Dict[str, object]] = []
        freq_rows: List[Dict[str, object]] = []
        for layer_type in ["gene", "pathway", "protein"]:
            sub = topk_long[topk_long["layer_type"] == layer_type].copy()
            if sub.empty:
                continue
            sets = []
            for rid in sorted(sub["run_id"].unique().tolist()):
                s = set(sub[sub["run_id"] == rid]["node"].astype(str).tolist())
                sets.append(s)
            jacc = _pairwise_jaccard(sets)
            st_rows.append(
                {
                    "layer_type": layer_type,
                    "n_ok_runs": int(len(sets)),
                    "pairwise_jaccard_mean": float(np.mean(jacc)) if jacc else 0.0,
                    "pairwise_jaccard_std": float(np.std(jacc)) if jacc else 0.0,
                    "pairwise_jaccard_min": float(np.min(jacc)) if jacc else 0.0,
                    "pairwise_jaccard_max": float(np.max(jacc)) if jacc else 0.0,
                }
            )

            # frequency + avg rank
            grp = sub.groupby("node").agg(freq=("run_id", "nunique"), avg_rank=("rank", "mean"), avg_score=("score", "mean"))
            grp = grp.sort_values(["freq", "avg_rank"], ascending=[False, True]).reset_index()
            grp["layer_type"] = layer_type
            freq_rows.extend(grp.to_dict(orient="records"))

        pd.DataFrame(st_rows).to_csv(join(os.getcwd(), exp_dir, "stability_jaccard.csv"), index=False)
        pd.DataFrame(freq_rows).to_csv(join(os.getcwd(), exp_dir, "node_frequency.csv"), index=False)

    return exp_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attribution stability experiment runner")
    p.add_argument("--model", required=True, help="model config name under configs/model/*.json (without .json)")
    p.add_argument("--dataset", required=True, help="dataset name under configs/dataset/*.json")
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=2024)
    p.add_argument("--seed-step", type=int, default=100)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--exp-root", default="results/attribution_stability")
    p.add_argument("--exp-id", default=None, help="optional fixed experiment id (folder name)")
    p.add_argument("--prepare-data", action="store_true", help="run prepare_data() + split_data() once before runs")
    p.add_argument(
        "--study-names",
        default="",
        help="comma-separated cBioPortal study IDs, e.g. prad_p1000 or brca_igr_2015,...",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    study_names = [s.strip() for s in str(args.study_names).split(",") if s.strip()] if args.study_names else None
    exp_dir = run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        n_runs=args.n_runs,
        seed_base=args.seed_base,
        seed_step=args.seed_step,
        topk=args.topk,
        exp_root=args.exp_root,
        exp_id=args.exp_id,
        prepare_data_flag=bool(args.prepare_data),
        study_names=study_names,
    )
    print(exp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
