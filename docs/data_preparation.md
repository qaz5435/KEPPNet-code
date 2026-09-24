# Data Preparation

This document specifies the cohort identifiers, file schemas, resource files,
and ordering used by the manuscript experiments. Patient identifiers must remain
unique, and the same split files must be reused by every compared method.

## 1. Public cancer cohorts

Download genomic and clinical files from [cBioPortal](https://www.cbioportal.org/).

| Cohort | Study identifiers | Manuscript sample count |
|---|---|---:|
| PRAD | `prad_p1000` | 1,013 |
| BRCA | `brca_igr_2015`, `brca_mbcproject_wagle_2017`, `brca_mbcproject_2022`, `brca_tcga_pan_can_atlas_2018` | 1,572 |

Do not substitute `prad_tcga_pan_can_atlas_2018` for `prad_p1000`; it is a
different study. Preserve the original cBioPortal sample identifiers while
joining mutation, CNA, clinical-label, and split tables.

Canonical raw and processed layout:

```text
data/structured/
|-- prad/
|   |-- raw_data/prad_p1000/
|   |   |-- data_mutations.txt
|   |   |-- data_cna.txt
|   |   `-- data_clinical_sample.txt
|   |-- processed/
|   |   |-- final_analysis_set_cross_important_only.csv
|   |   |-- data_CNA_paper.csv
|   |   `-- response_paper.csv
|   `-- splits/{training_set.csv,training_set_0.csv,validation_set.csv,test_set.csv}
`-- brca/
    |-- raw_data/<study identifier>/
    |-- processed/
    |   |-- final_analysis_set_cross_important_only.csv
    |   |-- data_CNA_paper.csv
    |   `-- response_paper.csv
    `-- splits/{training_set.csv,training_set_0.csv,validation_set.csv,test_set.csv}
```

The binary target convention is `0=primary` and `1=metastatic`. Target values
are used for training/evaluation only and must never be inserted into semantic
summary text.

## 2. Fixed 8:1:1 partitions

Prepare the two cohorts, then create stratified, non-overlapping partitions
with seed `422342`:

```bash
python -m src.data.processor.medical_data.prepare_data \
  --cancer_type prostate --study_names prad_p1000
python -m src.data.processor.medical_data.prepare_data \
  --cancer_type breast --study_names brca_igr_2015 \
  brca_mbcproject_wagle_2017 brca_mbcproject_2022 \
  brca_tcga_pan_can_atlas_2018

python -m src.data.processor.medical_data.split_data \
  --processed_dir data/structured/prad/processed \
  --output_dir data/structured/prad/splits \
  --seed 422342
python -m src.data.processor.medical_data.split_data \
  --processed_dir data/structured/brca/processed \
  --output_dir data/structured/brca/splits \
  --seed 422342
```

Never resplit data independently for competing models.

## 3. Reactome files

The code expects these files under `data/pathways/Reactome/`:

```bash
mkdir -p data/pathways/Reactome
wget https://reactome.org/download/current/ReactomePathways.txt \
  -O data/pathways/Reactome/ReactomePathways.txt
wget https://reactome.org/download/current/ReactomePathwaysRelation.txt \
  -O data/pathways/Reactome/ReactomePathwaysRelation.txt
wget https://reactome.org/download/current/ReactomePathways.gmt.zip \
  -O data/pathways/Reactome/ReactomePathways.gmt.zip
unzip -o data/pathways/Reactome/ReactomePathways.gmt.zip \
  -d data/pathways/Reactome/
```

Record the Reactome release number and access date used in an experiment. The
paper reports the hierarchy size before task-specific filtering; changing the
release can change that count and the resulting sparse masks.

## 4. STRING v12.0 files

Download the human (`9606`) network and protein information files from STRING:

```bash
mkdir -p data/protein
wget https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz
wget https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz
gunzip 9606.protein.links.full.v12.0.txt.gz 9606.protein.info.v12.0.txt.gz
mv 9606.protein.links.full.v12.0.txt 9606.protein.info.v12.0.txt data/protein/
```

## 5. Knowledge-graph relation tables

The full KG-enhanced configuration expects the following derived resources:

```text
data/genes/genes4prostate_cancer_2layer.csv
data/genes/genes4breast_cancer_2layer620.csv
data/genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt
data/pathways/Reactome/kg/prostate/genes2pathways_2hop_v1.csv
data/pathways/Reactome/kg/breast/genes2pathways_2hop_v1.csv
data/protein/prostate_id2id_2hop.csv
data/protein/breast_id2id_2hop.csv
```

`genes2pathways_2hop_v1.csv` contains `gene,pathway`; protein relation files
contain `child,parent`; selected-gene files contain a `genes` column. The HUGO
file is tab separated with chromosome, start, end, and gene-symbol columns and
is used to retain protein-coding genes. These are derived inputs to feature
selection and KG completion. Archive the exact versions used for reported
results and document their source, identifier mapping, filtering, and creation
date. They must be present for an exact run of the main configurations.

## 6. Semantic-input tables

Each `data/biolord_inputs/{prad,brca}/` directory must contain:

| File | Required columns | Purpose |
|---|---|---|
| `patient_master.csv` | `patient_id` | Defines profiled samples; other metadata are ignored by the text generator |
| `patient_mutated_genes.csv` | `patient_id`, `mutated_genes`, `amp_genes`, `del_genes`, optional count columns | Molecular events; lists are semicolon separated |
| `patient_top_pathways.csv` | `patient_id`, `top_pathway_names` | Pathway involvement computed from molecular alterations only |

Generate canonical summaries before defining experiment sets:

```bash
python scripts/prepare_data/build_patient_summaries.py --cancer_type prostate
python scripts/prepare_data/build_patient_summaries.py --cancer_type breast
```

The output `patient_summary.csv` contains only `patient_id`, `summary_text`, and
`cancer_type`. The summary generator rejects direct target/status and proxy
markers. It never consumes `study_source`, anatomical site, label, or split
values when constructing text.

## 7. Aligned experiment sets

```bash
python scripts/prepare_data/define_experiment_sets.py \
  --biolord_dir data/biolord_inputs \
  --output_dir data/biolord_inputs
```

This creates, for each cohort:

- `llm_text_set.csv`: valid summary, label, and split;
- `structured_set.csv`: valid mutation, CNA, label, and split; and
- `dual_strict_set.csv`: intersection used for branch comparison and fusion.

## 8. BioLORD-2023 embeddings

The official model identifier is `FremyCompany/BioLORD-2023`. To use an
auditable local snapshot, record its Hugging Face revision hash.

```bash
python scripts/prepare_data/build_biolord_embeddings.py \
  --cancer_type prostate \
  --set_name dual_strict_set \
  --model FremyCompany/BioLORD-2023 \
  --device cuda

python scripts/prepare_data/build_biolord_embeddings.py \
  --cancer_type breast \
  --set_name dual_strict_set \
  --model FremyCompany/BioLORD-2023 \
  --device cuda
```

Outputs are `embeddings.npy`, `embedding_patient_ids.csv`, and
`embedding_meta.csv`. The script aborts if any included sample lacks summary
text; it does not insert synthetic placeholder descriptions.
