from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import mapqc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.collections import PathCollection

import milopy.plot as milopl
import milopy.utils

from .metrics import DALogFC, MapQC, dist_uncertainty, prob_uncertainty

# Compatibility patch for packages expecting pandas<2 ``iteritems``.
pd.DataFrame.iteritems = pd.DataFrame.items



def _require_fields(adata: AnnData, celltype_key: str = "cell_type") -> None:
    if "oor_celltype" not in adata.uns:
        raise ValueError("`adata.uns['oor_celltype']` is required.")
    if "ref_query" not in adata.obs:
        raise ValueError("`adata.obs['ref_query']` is required.")
    if celltype_key not in adata.obs:
        raise ValueError(f"`adata.obs['{celltype_key}']` is required.")



def _rasterize_scatter_layers(fig: plt.Figure) -> None:
    for ax in fig.get_axes():
        for child in ax.get_children():
            if isinstance(child, PathCollection):
                child.set_rasterized(True)



def _save_rasterized_plot(fig: plt.Figure, output_path: Union[str, Path], dpi: int = 300) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _rasterize_scatter_layers(fig)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path



def _as_aligned_query_scores(adata: AnnData, key: str) -> np.ndarray:
    query_names = adata.obs_names[adata.obs["ref_query"] == "query"]
    scores = adata.uns[key]

    if isinstance(scores, pd.Series):
        return scores.reindex(query_names).to_numpy()
    return np.asarray(scores)



def _ensure_case_control_labels(
    adata: AnnData,
    celltype_key: str = "cell_type",
    ref_query_key: str = "ref_query",
    query_key: str = "query",
) -> None:
    target_celltype = adata.uns["oor_celltype"]
    query_mask = adata.obs[ref_query_key] == query_key

    adata.obs["case_control"] = "not_query"
    adata.obs.loc[query_mask, "case_control"] = "control"
    adata.obs.loc[query_mask & (adata.obs[celltype_key] == target_celltype), "case_control"] = "case"



def plot_prob_uncertainty(
    adata: AnnData,
    output_dir: Union[str, Path] = "./",
    celltype_key: str = "cell_type",
) -> Path:
    """Plot query-cell probabilistic uncertainty on UMAP."""
    _require_fields(adata, celltype_key=celltype_key)

    if "prob_uncertainty" not in adata.uns:
        print("→ Computing prob_uncertainty...")
        adata = prob_uncertainty(adata, label_key=celltype_key)
    else:
        print("→ Using pre-computed prob_uncertainty from adata.uns")

    print("→ Plotting prob_uncertainty...")
    adata_query = adata[adata.obs["ref_query"] == "query"].copy()
    adata_query.obs["prob_uncertainty"] = _as_aligned_query_scores(adata, "prob_uncertainty")

    sc.pl.umap(
        adata_query,
        color="prob_uncertainty",
        title="Prob_uncertainty",
        show=False,
        frameon=False,
        cmap="RdBu_r",
    )
    fig = plt.gcf()
    output_path = Path(output_dir) / "prob_uncertainty_plot.pdf"
    saved_path = _save_rasterized_plot(fig, output_path)
    print(f"✅ Saved: {saved_path.name}")
    return saved_path



def plot_dist_uncertainty(
    adata: AnnData,
    output_dir: Union[str, Path] = "./",
    celltype_key: str = "cell_type",
) -> Path:
    """Plot query-cell prototype-distance uncertainty on UMAP."""
    _require_fields(adata, celltype_key=celltype_key)

    if "dist_uncertainty" not in adata.uns:
        print("→ Computing dist_uncertainty...")
        adata = dist_uncertainty(adata, label_key=celltype_key)
    else:
        print("→ Using pre-computed dist_uncertainty from adata.uns")

    print("→ Plotting dist_uncertainty...")
    adata_query = adata[adata.obs["ref_query"] == "query"].copy()
    adata_query.obs["dist_uncertainty"] = _as_aligned_query_scores(adata, "dist_uncertainty")

    sc.pl.umap(
        adata_query,
        color="dist_uncertainty",
        title="Dist_uncertainty",
        show=False,
        frameon=False,
        cmap="RdBu_r",
    )
    fig = plt.gcf()
    output_path = Path(output_dir) / "dist_uncertainty_plot.pdf"
    saved_path = _save_rasterized_plot(fig, output_path)
    print(f"✅ Saved: {saved_path.name}")
    return saved_path



def plot_DAlogFC(
    adata: AnnData,
    output_dir: Union[str, Path] = ".",
    celltype_key: str = "cell_type",
    batch_key: str = "sample_id",
    **kwargs,
) -> Path:
    """Plot Milo neighborhood graph colored by DAlogFC statistics."""
    _require_fields(adata, celltype_key=celltype_key)

    if "nhood_adata" not in adata.uns:
        print("→ Computing DAlogFC...")
        adata = DALogFC(adata, celltype_key=celltype_key, batch_key=batch_key)
    else:
        print("→ Using pre-computed DAlogFC from adata.uns")

    print("→ Building neighborhood graph...")
    milopy.utils.build_nhood_graph(adata)

    print("→ Plotting DAlogFC...")
    plt.rcParams["figure.figsize"] = [10, 10]
    milopl.plot_nhood_graph(adata, show=False, **kwargs)

    fig = plt.gcf()
    output_path = Path(output_dir) / "DAlogFC_plot.pdf"
    saved_path = _save_rasterized_plot(fig, output_path)
    print(f"✅ Saved: {saved_path.name}")
    return saved_path



def plot_mapQC(
    adata: AnnData,
    output_dir: Union[str, Path] = ".",
    celltype_key: str = "cell_type",
    batch_key: str = "sample_id",
    mapqc_config: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Plot MapQC binary score visualization."""
    _require_fields(adata, celltype_key=celltype_key)

    if "mapqc_score" not in adata.obs:
        print("→ mapQC scores not found in adata.obs, computing...")
        cfg = dict(mapqc_config or {})
        MapQC(
            adata,
            embedding="X",
            ref_query_key="ref_query",
            ref_key="ref",
            query_key="query",
            study_key=cfg.get("study_key", "dataset_id"),
            batch_key=batch_key,
            n_nhoods=cfg.get("n_nhoods", 300),
            k_min=cfg.get("k_min", 1500),
            k_max=cfg.get("k_max", 10000),
            seed=cfg.get("seed", 10),
            overwrite=cfg.get("overwrite", True),
            return_nhood_info_df=False,
            return_sample_dists_to_ref_df=False,
        )
        _ensure_case_control_labels(adata, celltype_key=celltype_key)
        mapqc.evaluate(
            adata,
            case_control_key="case_control",
            case_cats=["case"],
            control_cats=["control"],
        )
    else:
        print("→ Using pre-computed mapQC scores from adata.obs")

    print("→ Plotting mapQC...")
    fig = mapqc.pl.umap.mapqc_scores_binary(adata, return_fig=True)
    output_path = Path(output_dir) / "mapQC_plot.pdf"
    saved_path = _save_rasterized_plot(fig, output_path)
    print(f"✅ Saved: {saved_path.name}")
    return saved_path



def plot_all_methods(
    adata: AnnData,
    output_dir: Union[str, Path] = "./",
    celltype_key: str = "cell_type",
    batch_key: str = "sample_id",
    mapqc_config: Optional[Mapping[str, Any]] = None,
    dalogfc_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Run and plot all supported methods in one call."""
    _require_fields(adata, celltype_key=celltype_key)

    print("=== Visualizing all methods ===")
    plot_prob_uncertainty(adata, output_dir=output_dir, celltype_key=celltype_key)
    plot_dist_uncertainty(adata, output_dir=output_dir, celltype_key=celltype_key)
    plot_DAlogFC(
        adata,
        output_dir=output_dir,
        celltype_key=celltype_key,
        batch_key=batch_key,
        **dict(dalogfc_kwargs or {}),
    )
    plot_mapQC(
        adata,
        output_dir=output_dir,
        celltype_key=celltype_key,
        batch_key=batch_key,
        mapqc_config=mapqc_config,
    )
    print("✅ All visualizations complete.")



def plot_auprc_summary(
    csv_path: Union[str, Path] = "evaluation_result.csv",
    output_dir: Union[str, Path] = "./",
) -> Path:
    """Plot AUPRC summary from ``evaluation_result.csv``."""
    df = pd.read_csv(csv_path, sep="\t", header=0)

    legacy_columns = ["Prob_uncertainty", "Dist_uncertainty", "DAlogFC", "mapQC"]
    if all(col in df.columns for col in legacy_columns):
        value_columns = legacy_columns
        rename_map = {col: col for col in value_columns}
    else:
        auprc_columns = [c for c in df.columns if c.endswith("_AUPRC")]
        if not auprc_columns:
            raise ValueError(
                "No AUPRC columns found. Expected legacy columns or '*_AUPRC' columns in CSV."
            )
        value_columns = auprc_columns
        rename_map = {col: col.replace("_AUPRC", "") for col in value_columns}

    df_melted = df.melt(
        id_vars=["adata_name"],
        value_vars=value_columns,
        var_name="Method",
        value_name="AUPRC",
    )
    df_melted["Method"] = df_melted["Method"].map(rename_map)

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df_melted, x="Method", y="AUPRC", palette="Set2", width=0.6, fliersize=0)
    sns.stripplot(data=df_melted, x="Method", y="AUPRC", color="black", size=4, jitter=True)

    plt.title("AUPRC Summary across Methods", fontsize=13)
    plt.ylabel("AUPRC", fontsize=11)
    plt.xlabel("")
    plt.ylim(0, 1.05)
    sns.despine()
    plt.tight_layout()

    output_path = Path(output_dir) / "AUPRC_summary_plot.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Saved: {output_path.name}")
    return output_path
