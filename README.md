# Graph Neural Network-Based Drug–Target Interaction Prediction for AML Drug Repurposing

This repository contains the implementation for an MSc Data Science dissertation on drug–target affinity (DTA) prediction and computational drug repurposing for acute myeloid leukaemia (AML). Version 2 refactors the verified version 1 workflow into reusable, testable source modules and thin notebooks that can run locally or in Google Colab.

> **Research-use statement:** AML rankings produced by this project are **computationally prioritised drug-repurposing candidates**. They are hypotheses for further investigation, not experimentally validated or clinically validated treatments. The software must not be used to guide patient care.

## Version and branch safety

- `v1-final` is the immutable historical reference implementation (frozen at commit `fd45653`). Do not move, retag, rewrite, or develop on it.
- All version 2 development belongs on `v2-refactor` or a later branch created from it.
- Generated experiments are not committed automatically. Review changes before committing or pushing.

Verify the working branch before making changes:

```bash
git branch --show-current
# expected: v2-refactor
```

The v1 implementation remains available without being copied or modified:

```bash
git show v1-final:README.md
git worktree add ../aml_gnn_dti_v1_read_only v1-final
```

Treat the optional worktree as read-only.

## Scientific scope

The project keeps Davis and KIBA as separate regression problems because their target values have different meanings.

| Dataset | Model target | Interpretation |
|---|---|---|
| Davis | `pKd = 9 - log10(Kd_nM)` | Higher values represent stronger binding on a pKd scale. |
| KIBA | Integrated KIBA score supplied by the source dataset | Higher values represent stronger interaction on the KIBA score scale. It is not pKd; this repository does not apply a new pKd conversion to it. |

The generic model output is therefore named **Predicted Binding Affinity Score**. Dataset-specific tables and figures must say either predicted Davis pKd or predicted KIBA score.

Three verified baseline families are compared fairly:

1. GCN + protein CNN
2. GAT + protein CNN
3. GIN + protein CNN

The advanced model adds a four-layer residual GINE drug encoder, frozen residue-level ESM-2 embeddings, a length-preserving protein CNN, bidirectional atom–residue cross-attention, masked pooling, and an MLP regression head. Frozen ESM-2 embeddings are intended to be cached once and reused.

Every architecture supports random, cold-drug, and cold-target evaluation. Model selection uses validation MSE; the test partition is reserved for final evaluation. Reported metrics are MSE, RMSE, MAE, concordance index, and Pearson correlation.

## Repository layout

```text
.
├── config.py                  # paths and reproducibility defaults
├── src/                       # reusable preprocessing, models, training and screening
├── notebooks/                 # ten thin local workflows
├── colab/                     # setup, baseline, advanced and screening workflows
├── data/
│   ├── raw/                   # untracked Davis/KIBA source files
│   ├── processed/             # untracked materialised datasets and ESM caches
│   ├── splits/                # full manifests local/Drive; compact metadata tracked
│   └── aml/                   # AML target and approved-drug metadata
├── models/                    # untracked checkpoints
├── results/
│   ├── metrics/               # lightweight experiment summaries
│   ├── predictions/           # full prediction tables are untracked
│   ├── plots/                 # selected dissertation figures
│   └── screening/             # lightweight AML ranking summaries
└── tests/                     # preprocessing, split, model and attention tests
```

Directory-specific README files document which artefacts belong in Git and which belong in local or Drive storage.

## Local setup

Python 3.10–3.12 is recommended. A CUDA-capable GPU is strongly recommended for ESM-2 embedding generation and advanced-model training, but preprocessing, tests, and small CPU smoke checks do not require one.

```bash
git clone https://github.com/Thebluesky303/msc-dissertation-drug-repurposing.git
cd msc-dissertation-drug-repurposing
git switch v2-refactor

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and PyTorch Geometric binary compatibility depends on the runtime and CUDA version. If the generic installation cannot load PyG operators, install the matching wheels using the official PyTorch Geometric installation instructions for the active PyTorch/CUDA build, then rerun the tests.

Run the lightweight test suite before experiments:

```bash
pytest -q
```

Open the workflow notebooks with:

```bash
jupyter lab
```

All notebooks default expensive `RUN_*` switches to `False`. Review the visible configuration cell, select one dataset/split/model/seed, and deliberately change only the required switch to `True`.

## Data setup

Raw benchmark files are deliberately excluded from Git. Place the GraphDTA-format files as follows:

```text
data/raw/
├── davis/
│   ├── ligands_can.txt
│   ├── proteins.txt
│   └── Y
└── kiba/
    ├── ligands_can.txt
    ├── proteins.txt
    └── Y
```

The preprocessing workflow preserves the verified v1 affinity transformations, source drug/target identifiers, atom features, protein truncation, and deterministic dataset ordering. Before any training run, execute notebook 01 with the v2 `prepare_dataset` API. It writes each materialised pair table in `data/processed/` together with a required `.metadata.json` sidecar recording source checksums, affinity semantics, and the processed-table checksum. Keep the CSV and sidecar together: legacy processed CSVs without this provenance are intentionally rejected rather than silently trusted.

Full pair-membership manifest CSVs are generated artefacts and remain local or in Drive; the complete 18-manifest grid is too large for the lightweight Git policy. Git may retain their compact `.metadata.json` sidecars, including dataset/pair fingerprints, seed, ratios, counts, leakage diagnostics, and manifest checksum. Deterministic split code plus the exact dataset ordering and metadata reconstruct the full manifests.

Splits use seeds `42`, `123`, and `2026`, with seed `42` as the default single-run development seed. Cold-drug and cold-target partitions are disjoint by source identifier. Identifier-disjointness does not necessarily imply chemical-structure or amino-acid-sequence disjointness; leakage diagnostics should be reviewed alongside each manifest.

Never combine Davis and KIBA target values into one regression dataset.

For AML screening, retain the tracked target panel in `data/aml/aml_targets.csv` and provide `data/aml/approved_drugs_raw.csv` locally or through Drive. The approved-drug input requires `drug_id`, `drug_name`, and `smiles`; preserve approval status, source, version, and provenance columns whenever the supplier provides them. The preparation step writes the deduplicated model-ready metadata separately and records rejected rows with reasons.

## Notebook workflow

Run only the stages needed for the current question:

1. `notebooks/01_data_preparation.ipynb` — validate source data, preprocess pairs, and materialise deterministic split manifests.
2. `notebooks/02_davis_baselines.ipynb` — configure one Davis GCN/GAT/GIN baseline run.
3. `notebooks/03_davis_model_comparison.ipynb` — compare lightweight Davis experiment records using validation evidence.
4. `notebooks/04_cold_start_evaluation.ipynb` — inspect cold-drug/cold-target integrity and generalisation summaries.
5. `notebooks/05_kiba_experiments.ipynb` — configure one KIBA baseline run.
6. `notebooks/06_baseline_model_selection.ipynb` — select a baseline without using test metrics for selection.
7. `notebooks/07_advanced_gine_esm2.ipynb` — cache frozen residue embeddings and configure one advanced run.
8. `notebooks/08_baseline_vs_advanced.ipynb` — compare matched baseline and advanced experiment records.
9. `notebooks/09_aml_approved_drug_screening.ipynb` — prepare an approved small-molecule library and explicitly run two-model AML inference.
10. `notebooks/10_final_candidate_comparison.ipynb` — compare ranks, overlap, and target-specific agreement.

Training, ESM-2 generation, and screening never start merely by opening or running a notebook with its default settings.

## Google Colab and Drive

The `colab/` notebooks separate durable and temporary storage:

```text
Google Drive (durable)
└── MyDrive/msc-dissertation-drug-repurposing/
    ├── data/raw/{davis,kiba}/
    ├── data/aml/
    ├── data/splits/manifests/   # full membership CSVs + metadata sidecars
    ├── caches/esm2/
    ├── checkpoints/
    └── results/

Colab runtime (temporary, faster)
├── /content/msc-dissertation-drug-repurposing/   # v2-refactor clone
└── /content/dti_workspace/                       # copied data and generated artefacts
```

The Drive root and Git URL are visible parameters, not hard-coded assumptions. The Colab flow is:

1. clone or fast-forward `v2-refactor`;
2. install dependencies;
3. mount Google Drive;
4. copy only required input data/cache files into `/content`;
5. deliberately enable one configured experiment;
6. copy checkpoints, metrics, selected summaries, and caches back to Drive.

Use `colab/01_colab_setup.ipynb` first. The remaining notebooks independently guard data copying, training, screening, and Drive synchronisation behind `RUN_* = False` switches. Colab runtimes are ephemeral: a successful local output is not durable until its sync step completes.

## Experiment records and reproducibility

Each completed run should record:

- dataset, split, model, protein encoder, and seed;
- best epoch and best validation MSE;
- final test MSE, RMSE, MAE, concordance index, and Pearson correlation;
- trainable parameter count and runtime;
- checkpoint path;
- the full serialisable configuration used.

Use validation MSE to select checkpoints and architectures. Do not tune on or select by test performance. Full split manifests, per-pair prediction tables, checkpoints, and ESM-2 caches stay outside Git; only compact manifest metadata, summaries, and selected figures should be versioned.

## AML screening interpretation

The AML application crosses a deduplicated approved small-molecule library with the established AML target panel and predicts every compatible pair with:

- the selected baseline model; and
- the advanced GINE + frozen ESM-2 + cross-attention model.

Outputs include top-10 and top-20 overall lists, top-five candidates per target, baseline/advanced scores, ranks, rank differences, overlap, and target-specific agreement. Canonical SMILES and InChIKey are used where available to remove invalid, duplicate, multi-component, or graph-incompatible records.

These rankings describe computational prioritisation only. This v2 workflow intentionally excludes the large ChEMBL/PubMed/ClinicalTrials “validation” pipeline used experimentally in v1; database mentions or rank agreement do not establish efficacy, safety, mechanism, or clinical validity.

Provenance-safe screening accepts checkpoints written by the standard v2 experiment runners. Legacy v1 weights without the v2 configuration, data, split, and cache provenance remain historical artefacts and are intentionally rejected. No automatic migration is provided; any future migration would require a separate, explicit, audited process.

## Licence and data provenance

No licence is asserted here for third-party datasets or pretrained model weights. Users are responsible for complying with the Davis, KIBA/GraphDTA, approved-drug source, UniProt, ESM-2, RDKit, PyTorch, and PyTorch Geometric terms and for recording the exact source version/checksum used in each reproducible run.
