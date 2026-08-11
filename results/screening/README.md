# AML screening outputs

Keep compact ranking and comparison summaries here. Full drug–target prediction matrices, caches, and embedding tensors are excluded from Git and should be retained in durable Drive storage.

Retain `screening_provenance.json` with each reviewed screen. It should embed the standard v2 runner provenance and record the resolved path, byte size, and chunked SHA-256 for the raw approved-drug CSV, AML targets CSV, residue-cache tensor, and its `.metadata.json` sidecar.

All candidate tables must be described as **computationally prioritised drug-repurposing candidates**, not experimentally or clinically validated treatments.
