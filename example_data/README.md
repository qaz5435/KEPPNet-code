# Synthetic Example Data

This directory contains synthetic records for a lightweight pipeline check. It
does not contain real participants, patient-level cBioPortal exports, or any data
used to obtain the manuscript results.

The source cohorts analyzed in the manuscript are publicly available through
cBioPortal, but their patient-level files are not duplicated in this code
repository. Download them directly from the source studies and follow the source
terms of use.

## Included smoke-test files

- `sample_mutation.csv`: rows are synthetic sample IDs, columns are gene symbols,
  and values are binary mutation indicators.
- `sample_cna.csv`: rows are synthetic sample IDs, columns are gene symbols, and
  values encode deletion (`-1`), neutral (`0`), or amplification (`1`).
- `sample_response.csv`: synthetic labels with `0=primary` and `1=metastatic`.

Run:

```bash
python scripts/run_demo.py
```

The resulting metrics are a software smoke test and must not be reported as
scientific results.

## Real-data identifiers

- PRAD: `prad_p1000`
- BRCA: `brca_igr_2015`, `brca_mbcproject_wagle_2017`,
  `brca_mbcproject_2022`, `brca_tcga_pan_can_atlas_2018`

See [`../docs/data_preparation.md`](../docs/data_preparation.md) for full schemas,
resource downloads, and leakage safeguards.
