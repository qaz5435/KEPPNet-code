# Data directory

Patient-level cohort files, derived knowledge tables, pretrained embeddings,
and model outputs are intentionally not committed. Create the following layout
after obtaining the resources listed in `docs/data_preparation.md`:

```text
data/
|-- structured/{prad,brca}/{raw_data,processed,splits}/
|-- biolord_inputs/{prad,brca}/
|-- pathways/Reactome/
|-- protein/
`-- genes/
```

The main model requires the cohort-specific selected-gene lists and two-hop KG
tables documented in the data-preparation guide. Do not upload patient-level
data or credentials to a public repository.
