# Processed data

Generated pair tables, graph tensors, and frozen ESM-2 residue caches live here during local runs. They are reproducible and potentially large, so they are excluded from Git and should be stored in Google Drive for long-running Colab work.

Do not manually edit materialised pairs. Run notebook 01 with the v2 `prepare_dataset` API and regenerate them from the raw files and recorded configuration. Every processed CSV must stay beside its `.metadata.json` source/affinity/checksum sidecar; downstream training intentionally rejects legacy CSVs without that sidecar.
