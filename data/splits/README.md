# Split reproducibility artefacts

This directory separates full pair-membership manifests from compact, versionable provenance:

- `manifests/*.csv`: full ordered pair membership; generated locally or in Colab and retained in Drive, not Git;
- `manifests/*.metadata.json`: compact sidecars containing dataset/pair fingerprints, seed, ratios, counts, leakage diagnostics, manifest checksum, and schema/version information;
- `metadata/`: other compact dataset/split provenance JSON when needed.

The full 18-manifest experiment grid is not a lightweight Git artefact. Reconstruct it with the deterministic split code, the exact processed-dataset ordering, and recorded metadata, then verify its checksum against the sidecar before training.

Cold-drug and cold-target checks operate on preserved source identifiers. Chemical-structure and protein-sequence duplication diagnostics should also be recorded because identifier-disjointness is not the same as strict structure/sequence disjointness.
