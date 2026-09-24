# Reproducibility and Reporting Notes

This repository is organized to match the accompanying KEPPNet manuscript.
Use the following checklist before producing or releasing reported results.

## Cohorts and partitions

- PRAD source: `prad_p1000`.
- BRCA sources: `brca_igr_2015`, `brca_mbcproject_wagle_2017`,
  `brca_mbcproject_2022`, and `brca_tcga_pan_can_atlas_2018`.
- Preserve sample identifiers and verify that no sample occurs in more than one
  partition.
- Use the same stratified 8:1:1 partition for KEPPNet and every baseline.
- Default split seed: `422342`.

## Semantic-input audit

Only molecularly derived content may appear in `summary_text`. Before encoding,
inspect the generated file and confirm that it contains none of the following:

- target/status words such as `primary`, `metastatic`, or `metastasis`;
- anatomical or collection site;
- study/source identifiers;
- clinical outcomes;
- train/validation/test membership; or
- any field derived from the prediction label.

The summary generator implements an explicit forbidden-marker check and writes
labels/splits only to separate set and metadata files.

## Training and selection

- Fit preprocessing transformations on the training partition only.
- Compute class weights from training labels only.
- Keep the BioLORD encoder frozen.
- Select the fusion weight using validation AUC only.
- Keep the decision threshold fixed at `0.5` for the canonical analysis.
- Never use test labels for model selection, early stopping, fusion selection,
  preprocessing, or threshold selection.
- Use `configs/model/prad-main.json` and `configs/model/brca-main.json` for the
  manuscript main structured configurations.

## External-resource provenance

For each experiment, archive or record:

- cBioPortal study identifiers and download date;
- Reactome release number and access date;
- STRING version (`v12.0`) and file names;
- BioLORD model identifier and snapshot revision;
- identifier-mapping procedure;
- KG source, extraction date, relation types, and two-hop filtering rules; and
- hashes of derived relation tables and final split files.

The selected-gene lists and KG-derived tables are essential to reproducing the
full KG-enhanced model. The main configurations stop with an explicit error if
the KG tables are absent. Use a named `*-no-kg` configuration only when the
direct-evidence ablation is intended; do not present it as the full model.

## Result reporting

- Report whether a table entry is a single fixed-seed result or a mean across
  repeated seeds.
- Keep Accuracy, Precision, Recall, F1, AUPRC, and AUC definitions identical
  across models.
- Report test-set metrics only after all validation decisions are fixed.
- Treat attribution and perturbation results as evidence of model behavior, not
  proof of biological causality.

## Automated repository check

Run before tagging a release:

```bash
python scripts/check_repository_consistency.py
python -m compileall -q scripts src
```
