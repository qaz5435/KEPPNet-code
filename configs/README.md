# KEPPNet Configuration Files

## configs/model/

Model configs are loaded by `src/config/configuration.py` from `configs/model/<name>.json`.
Pass the config name (without `.json`) via `--model_config` or use the defaults in `scripts/train.py`.

### Naming conventions

| Prefix | Meaning |
|--------|---------|
| `*-main.json` | Paper main experiment config — reproduces the primary results reported in the paper |
| `*-no-kg.json` | Ablation: KG augmentation disabled (`use_kg: false`) |
| `*-pure-reactome.json` | Ablation: pure Reactome hierarchy, no protein insertion |
| `*-reactome-protein.json` | Ablation: Reactome + PPI hybrid, no KG augmentation |
| `demo-*.json` | Public demo configs — not tuned for best performance, designed for quick onboarding |

### Main experiment configs

| File | Cancer | Network | Layers | Batch size | Epochs | Notes |
|------|--------|---------|--------|-----------:|-------:|-------|
| `brca-main.json` | BRCA | reactome_protein_kg | 3 | 50 | 300 | Manuscript main configuration |
| `prad-main.json` | PRAD | reactome_protein_kg | 4 | 50 | 300 | Manuscript main configuration |

### Ablation configs

| File | Cancer | Network | KG |
|------|--------|---------|-----|
| `brca-no-kg.json` | BRCA | reactome_protein_kg | disabled |
| `brca-pure-reactome.json` | BRCA | reactome | — |
| `brca-reactome-protein.json` | BRCA | reactome_protein | — |
| `prad-no-kg.json` | PRAD | reactome_protein_kg | disabled |
| `prad-pure-reactome.json` | PRAD | reactome | — |
| `prad-reactome-protein.json` | PRAD | reactome_protein | — |

### Demo configs

| File | Purpose |
|------|---------|
| `demo-minimal.json` | Minimal pipeline check — pure Reactome, 50 epochs, no PPI/KG data needed |
| `demo-fast.json` | Fast exploration — KG-enhanced network, 100 epochs, relaxed thresholds |
| `demo-dual-branch.json` | Dual-branch pipeline demo — structural branch settings for PathProNet + BioLORD fusion |

## configs/dataset/

| File | Description |
|------|-------------|
| `breast_kg.json` | BRCA dataset config (multi-study) |
| `prostate_kg.json` | PRAD dataset config |

The PRAD cohort identifier is `prad_p1000`; `prostate_kg` is only the local
configuration name. Model and dataset JSON files are loaded from this root-level
`configs/` directory by `src/config/configuration.py`.

## Usage examples

```bash
# Train with paper main config (default)
python scripts/train.py --mode pathpronet --cancer_type prostate

# Train with a specific config
python scripts/train.py --mode pathpronet --cancer_type breast --model_config brca-main

# Run ablation
python scripts/train.py --mode pathpronet --cancer_type prostate --model_config prad-no-kg

# Quick pipeline check (no PPI/KG data required)
python scripts/train.py --mode pathpronet --cancer_type prostate --model_config demo-minimal
```
