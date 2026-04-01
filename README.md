# mapAD

mapAD (Map Anomaly Detection) is a benchmarking toolkit for abnormal / out-of-reference (OOR) state detection in single-cell reference mapping.

## Scope

mapAD provides a unified evaluation pipeline for four methods:

- `Prob_uncertainty`
- `Dist_uncertainty`
- `DAlogFC`
- `mapQC`

It reports:

- AUPRC
- AUROC
- TPR@10%FDR
- FPR@10%FDR
- Actual FDR
- Actual Precision
- Decision Threshold

## Installation

> Keep the original installation sequence unchanged.

1. Configure Conda environment

```bash
conda create -n mapbench python=3.10 && y && conda activate mapbench
```

2. Install required dependencies

```bash
conda install r-base=4.3.1
conda install compilers
conda install -c conda-forge xz zlib
conda install bioconda::bioconductor-edger=4.0.16
```

3. Compile and install `milopy`

```bash
cd milopy
pip install .
```

4. Install `mapAD`

```bash
cd ..
pip install .
```

## Required AnnData fields

Before running evaluation, make sure your `adata` has:

- `adata.obs['ref_query']` with `ref` / `query`
- `adata.obs[celltype_key]` (for example: `cell_type`)
- `adata.obs[batch_key]` (for example: `sample_id`)
- `adata.uns['oor_celltype']` (target abnormal cell type)

Example:

```python
adata.obs["ref_query"]
adata.obs["cell_type"]
adata.obs["sample_id"]
adata.uns["oor_celltype"] = "Disease_Tcell"
```

## Workflow

### 1) Load data and set config

```python
import scanpy as sc
from mapad.run_evaluation import run_evaluation

adata = sc.read_h5ad("/path/to/your_data.h5ad")

celltype_key = "cell_type"
batch_key = "sample_id"

mapqc_config = {
    "n_nhoods": 300,
    "k_min": 1500,
    "k_max": 10000,
    "study_key": "dataset",
    "seed": 10,
}
```

### 2) Run full evaluation

```python
adata = run_evaluation(
    adata_path=adata,
    celltype_key=celltype_key,
    batch_key=batch_key,
    mapqc_config=mapqc_config,
    output_dir="./results",
    target_fdr=0.1,
)
```

Main output:

- `./results/evaluation_result.csv` (tab-separated)

### 3) Optional: save raw scores and compute metrics later

```python
from mapad.run_evaluation import run_prediction_step, compute_metrics_from_file

score_file = run_prediction_step(
    adata_path=adata,
    celltype_key=celltype_key,
    batch_key=batch_key,
    mapqc_config=mapqc_config,
    output_file="./results/raw_scores.csv",
)

metrics = compute_metrics_from_file(score_file, target_fdr=0.1)
print(metrics)
```

### 4) Generate plots

```python
from mapad.run_visualization import plot_all_methods, plot_auprc_summary

plot_all_methods(
    adata,
    output_dir="./figures",
    celltype_key=celltype_key,
    batch_key=batch_key,
    mapqc_config=mapqc_config,
)

plot_auprc_summary(
    csv_path="./results/evaluation_result.csv",
    output_dir="./figures",
)
```

## Main API

From `mapad.run_evaluation`:

- `run_evaluation(...)`
- `run_evaluation_for_auprc(...)`
- `run_prediction_step(...)`
- `compute_metrics_from_file(...)`

From `mapad.run_visualization`:

- `plot_prob_uncertainty(...)`
- `plot_dist_uncertainty(...)`
- `plot_DAlogFC(...)`
- `plot_mapQC(...)`
- `plot_all_methods(...)`
- `plot_auprc_summary(...)`

## Output files

Typical outputs:

- `evaluation_result.csv`
- `raw_scores.csv` (optional)
- `prob_uncertainty_plot.pdf`
- `dist_uncertainty_plot.pdf`
- `DAlogFC_plot.pdf`
- `mapQC_plot.pdf`
- `AUPRC_summary_plot.pdf`

## Project layout

```text
mapAD/
├─ mapad/
│  ├─ metrics/
│  │  ├─ prob_uncertainty.py
│  │  ├─ dist_uncertainty.py
│  │  ├─ dalogfc.py
│  │  └─ mapqc.py
│  ├─ evaluation/
│  │  └─ identification_metrics.py
│  ├─ run_evaluation.py
│  └─ run_visualization.py
└─ milopy/
```

## Notes

- `milopy` is required by the DAlogFC workflow.
- mapAD evaluates mapping outputs; it does not train reference-mapping models.
