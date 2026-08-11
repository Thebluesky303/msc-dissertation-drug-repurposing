# Experiment metrics

Completed runners write one strict JSON record per configuration and append a
compact row to `experiment_log.csv`.  Validation MSE selects checkpoints; test
metrics are recorded only after the selected checkpoint is reloaded.

Small final comparison tables may be versioned after review.  Full histories
and repeated development outputs should remain local or in durable Drive
storage.
