# KEPPNet Model Architecture

KEPPNet classifies the status of already profiled primary and metastatic tumor
samples. It combines an explicit pathway--protein structural prior with a frozen
biomedical semantic representation of the same genomic alterations.

## Structured branch

### Input

For the selected gene set, three binary channels are aligned by sample and gene:

1. somatic mutation;
2. copy-number deletion; and
3. copy-number amplification.

The concatenated alteration vector is passed through a gene-grouping layer and
then through masked sparse layers. Exact dimensions are cohort- and release-
dependent and should be read from the generated network manifest rather than
hard-coded.

### Biological masks

- **Reactome hierarchy:** supplies parent--child pathway connectivity.
- **PPI-supported routing:** inserts protein mediators between related pathway
  levels when supported by STRING evidence.
- **KG completion:** adds two-hop protein--pathway/protein evidence only when
  direct bridge evidence is insufficient.
- **Hierarchy-aware filtering:** uses `min_shared_proteins`,
  `min_shared_ratio`, and `use_dynamic_threshold` to control sparsity.
- **Hub control:** uses `hub_threshold` and `max_hub_proteins` to prevent
  ubiquitous proteins from dominating the network.
- **Mediator cap:** `max_mediators_per_edge` bounds each retained pathway pair
  after shared-protein, cross-set STRING, and optional KG evidence are combined.

Each allowable connection is encoded by a fixed mask. Intermediate layers have
auxiliary prediction heads, and their binary cross-entropy losses are combined
using the configured `loss_weights`. At inference, the structured probability
is the arithmetic mean of all auxiliary-head probabilities; equivalently, the
normalized layer-contribution coefficients are uniform and sum to one.

### Main structured configurations

| Cohort | Config | Hidden hierarchy depth | Batch size | Epochs | Initial LR |
|---|---|---:|---:|---:|---:|
| PRAD | `configs/model/prad-main.json` | 4 | 50 | 300 | 0.001 |
| BRCA | `configs/model/brca-main.json` | 3 | 50 | 300 | 0.001 |

Both use Adam, L2 coefficient `0.001`, and a learning-rate multiplier of `0.25`
every 50 epochs. Class weights are computed from the training partition.

## Semantic branch

### Label-free text

`scripts/prepare_data/build_patient_summaries.py` uses a common deterministic
template for both cohorts. A typical structure is:

```text
A {cancer type} patient.
Somatic mutations affect {count} genes, including {gene list}.
Copy-number amplification/deletion affects {counts} genes, including {gene lists}.
The molecular profile contains {count} recorded alteration events across channels.
The most involved biological pathways include {molecularly derived pathway names}.
```

The encoded text excludes clinical labels, the words primary/metastatic,
anatomical site, study/source identifiers, and split membership. The output CSV
also omits label and split columns so the encoding step cannot accidentally pass
them to BioLORD.

### Frozen encoder and classifier

The model identifier is `FremyCompany/BioLORD-2023`. Its pretrained parameters
remain frozen. A lightweight classifier maps the 768-dimensional embedding to a
single logit:

```text
h_sem -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid
```

Main classifier settings used by the dual-branch runner are:

| Cohort | Hidden dimension | Dropout | Learning rate | Batch size |
|---|---:|---:|---:|---:|
| PRAD | 128 | 0.5 | 0.0003 | 32 |
| BRCA | 128 | 0.3 | 0.0010 | 32 |

The class weight is the negative-to-positive ratio computed from training data.
The scaler is fit on the training partition only.

## Decision-level fusion

```text
p_final = lambda * p_structured + (1 - lambda) * p_semantic
```

The canonical implementation is `scripts/train/train_dual_branch.py`.
`lambda` is searched over `{0.0, 0.1, ..., 1.0}` and selected by validation AUC.
The selected value is fixed for test evaluation. The classification threshold is
fixed at `0.5`; it is not tuned on the test set. The standalone
`scripts/evaluate/eval_late_fusion.py` applies the same rule to previously
exported branch probabilities.

## Feature-level fusion ablation

`scripts/evaluate/eval_feature_fusion.py` concatenates frozen structured and
semantic representations and trains only the fusion MLP. This is an ablation and
is not the main KEPPNet result.

## Interpretability

The structured branch supports DeepLIFT, Integrated Gradients, and GradientSHAP.
Repeated-seed summaries identify entities that recur across methods and runs.
Attribution-guided removal is compared with degree-matched random removal to
assess model reliance. These analyses support interpretation of the trained
predictor but do not independently establish biological causality.
