# Model checkpoints

Model weights and optimiser/training state are generated artefacts and are excluded from Git. Store long-lived checkpoints in Google Drive and record their path plus configuration in the experiment log.

Checkpoint selection must use validation MSE. Load the selected checkpoint only once for final test evaluation or downstream AML screening.

Provenance-safe screening accepts standard v2 runner checkpoints. Legacy v1 weights that lack v2 configuration, data, split, or residue-cache provenance are historical and intentionally rejected. No default migration path is provided; migration would need a separate, explicit, audited process.
