# Elliptic inputs

The three raw CSVs are copied here locally and ignored by Git because their combined size is about 665 MiB. To reproduce a fresh checkout, restore these same files in this directory:

- `elliptic_txs_features.csv`: transaction ID, time step and 165 numerical features; no header.
- `elliptic_txs_classes.csv`: transaction ID and class (1 illicit, 2 licit, unknown).
- `elliptic_txs_edgelist.csv`: directed transaction edges.

`manifest.json` records the exact input sizes and SHA-256 hashes used by the experiments. This repository does not bundle data into Git history or assign a new license to the dataset. The notebook reads these local files without consulting the original source directory.
