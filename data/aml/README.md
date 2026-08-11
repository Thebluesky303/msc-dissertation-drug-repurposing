# AML screening inputs

`aml_targets.csv` is the established six-target AML panel retained from the verified v1 workflow. Preserve its target identifiers, UniProt-derived sequences, metadata, and provenance unless a deliberate scientific change is reviewed and documented.

The final v2 screen requires a substantially larger approved small-molecule source table than the legacy 12-drug exploratory panel. Supply `approved_drugs_raw.csv` locally or through Google Drive with at least:

- `drug_id`: stable identifier from the source;
- `drug_name`: preferred drug name;
- `smiles`: structure supplied by the source;
- approval status and jurisdiction where available;
- source database/reference and provenance fields where available.

Record the source name, licence/terms, release or retrieval date, version, file checksum, and any inclusion filter. Do not download data implicitly from a notebook. The preparation workflow canonicalises structures, calculates InChIKey where supported, removes invalid/duplicate/multi-component or graph-incompatible records, and writes an auditable rejection table.

The 12-drug v1 panel may be retained only as a smoke-test/reference fixture; it is not the final approved-drug screening library.

All model outputs are **computationally prioritised drug-repurposing candidates**, not experimentally or clinically validated treatments.
