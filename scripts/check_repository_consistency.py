#!/usr/bin/env python
"""Run lightweight paper--repository consistency checks without patient data."""

from __future__ import annotations

import importlib.util
import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_summary_module():
    path = ROOT / "scripts/prepare_data/build_patient_summaries.py"
    spec = importlib.util.spec_from_file_location("build_patient_summaries", path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_configs() -> None:
    for name, layers in (("prad-main", 4), ("brca-main", 3)):
        path = ROOT / f"configs/model/{name}.json"
        require(path.exists(), f"Missing main configuration: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
        require(config.get("epoch") == 300, f"{name}: epoch must be 300")
        require(config.get("batch_size") == 50, f"{name}: batch_size must be 50")
        require(config.get("lr") == 0.001, f"{name}: lr must be 0.001")
        require(config.get("penalty") == 0.001, f"{name}: penalty must be 0.001")
        require(config.get("n_hidden_layers") == layers, f"{name}: unexpected layer depth")
        require(config.get("max_f1") is False, f"{name}: evaluation threshold must remain fixed")

    datasets = {
        "prostate_kg": (
            "data/structured/prad/processed",
            "data/structured/prad/splits",
            "genes4prostate_cancer_2layer.csv",
        ),
        "breast_kg": (
            "data/structured/brca/processed",
            "data/structured/brca/splits",
            "genes4breast_cancer_2layer620.csv",
        ),
    }
    for name, expected in datasets.items():
        path = ROOT / f"configs/dataset/{name}.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        observed = (config.get("processed_dir"), config.get("splits_dir"), config.get("selected_genes"))
        require(observed == expected, f"{name}: unexpected data paths or selected-gene file")


def check_label_free_summary() -> None:
    module = load_summary_module()
    molecular = {
        "mutated_genes": "TP53;PTEN",
        "amp_genes": "MYC",
        "del_genes": "RB1",
        "num_mutated_genes": "2",
        "num_amp_genes": "1",
        "num_del_genes": "1",
        # These metadata fields intentionally must be ignored.
        "study_source": "brca_mbcproject_2022",
        "metastatic_site": "liver",
        "label": "1",
        "split": "test",
    }
    text = module.build_summary_text(
        "breast cancer",
        molecular,
        {"top_pathway_names": "DNA Repair;Cell Cycle"},
    )
    lowered = text.lower()
    for marker in module.FORBIDDEN_TEXT_MARKERS:
        require(marker not in lowered, f"Forbidden marker leaked into summary: {marker}")
    require("TP53" in text and "DNA Repair" in text, "Molecular content missing from summary")


def check_documentation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    data_doc = (ROOT / "docs/data_preparation.md").read_text(encoding="utf-8")
    combined = readme + "\n" + data_doc
    require("prad_p1000" in combined, "Official PRAD identifier missing")
    require("github.com/qaz5435/KEPPNet-code" in readme, "Repository URL is incorrect")
    forbidden_placeholders = ("your-username", "[Authors]", "[Journal]")
    for token in forbidden_placeholders:
        require(token not in combined, f"Placeholder remains in documentation: {token}")
    require(
        "cancer treatment response classification" not in readme.lower(),
        "README still describes the wrong prediction task",
    )


def check_known_obsolete_references() -> None:
    obsolete = (
        "prostate-network_reactome_protein_kg",
        "breast-network_reactome_protein_kg",
        "prad-best",
        "brca-best",
    )
    scanned = [ROOT / "README.md"]
    scanned.extend(
        path for path in (ROOT / "scripts").rglob("*.py")
        if path.name != "check_repository_consistency.py"
    )
    scanned.extend((ROOT / "src").rglob("*.py"))
    for path in scanned:
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in obsolete:
            require(token not in text, f"Obsolete config reference {token!r} in {path.relative_to(ROOT)}")


def check_local_import_targets() -> None:
    """Catch imports of repository modules that do not exist on disk."""
    for path in list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if not (module.startswith("src.") or module.startswith("scripts.")):
                    continue
                target = ROOT.joinpath(*module.split("."))
                require(
                    target.with_suffix(".py").exists() or (target / "__init__.py").exists(),
                    f"Missing local import target {module!r} referenced by {path.relative_to(ROOT)}",
                )


def check_no_legacy_data_swapping() -> None:
    scanned = [
        path for path in (list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "src").rglob("*.py")))
        if path.name != "check_repository_consistency.py"
    ]
    forbidden = ("swap_brca", "data/breast/processed", "genes4beast")
    for path in scanned:
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in forbidden:
            require(token not in text, f"Legacy data-layout token {token!r} in {path.relative_to(ROOT)}")


def main() -> int:
    checks = (
        check_configs,
        check_label_free_summary,
        check_documentation,
        check_known_obsolete_references,
        check_local_import_targets,
        check_no_legacy_data_swapping,
    )
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")
    print("Repository consistency checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
