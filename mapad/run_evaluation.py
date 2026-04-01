from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple, Union

import mapqc
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from .evaluation import compute_identification_metrics
from .metrics import DALogFC, MapQC, dist_uncertainty, prob_uncertainty

# Compatibility patch for packages expecting pandas<2 ``iteritems``.
pd.DataFrame.iteritems = pd.DataFrame.items

AnnDataInput = Union[str, Path, AnnData]
METHOD_ORDER = ("Prob_uncertainty", "Dist_uncertainty", "DAlogFC", "mapQC")



def _load_adata(adata_input: AnnDataInput) -> Tuple[AnnData, str]:
    if isinstance(adata_input, AnnData):
        return adata_input, "unnamed_adata"

    adata_path = Path(adata_input)
    adata = sc.read_h5ad(adata_path)
    return adata, adata_path.stem



def _validate_adata_inputs(adata: AnnData, celltype_key: str) -> None:
    if "oor_celltype" not in adata.uns:
        raise KeyError("Missing `oor_celltype` in adata.uns")
    if "ref_query" not in adata.obs:
        raise KeyError("Missing `ref_query` in adata.obs")
    if celltype_key not in adata.obs:
        raise KeyError(f"Missing `{celltype_key}` in adata.obs")



def _prepare_case_control_labels(
    adata: AnnData,
    celltype_key: str,
    ref_query_key: str = "ref_query",
    query_key: str = "query",
) -> Tuple[str, pd.Series]:
    target_celltype = adata.uns["oor_celltype"]

    query_mask = adata.obs[ref_query_key] == query_key
    adata.obs["case_control"] = "not_query"
    adata.obs.loc[query_mask, "case_control"] = "control"

    target_mask = query_mask & (adata.obs[celltype_key] == target_celltype)
    adata.obs.loc[target_mask, "case_control"] = "case"
    return target_celltype, query_mask



def _to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        return values.to_numpy().ravel()
    if isinstance(values, pd.Series):
        return values.to_numpy().ravel()
    return np.asarray(values).ravel()



def _extract_query_scores(adata: AnnData, score_key: str, query_mask: pd.Series) -> np.ndarray:
    scores = adata.uns[score_key]
    if isinstance(scores, pd.Series):
        return scores.reindex(adata.obs_names[query_mask]).to_numpy()
    return _to_numpy(scores)



def _compute_dalogfc_targets(
    adata: AnnData,
    celltype_key: str,
    target_celltype: str,
) -> np.ndarray:
    adata.obs["is_abnormal"] = (adata.obs[celltype_key] == target_celltype).astype(int)

    sample_adata = adata.uns["sample_adata"]
    groups_mat = sample_adata.varm["groups"].copy()

    abnormal_mask = adata.obs["is_abnormal"] == 1
    oor_members = groups_mat[:, abnormal_mask]

    if hasattr(oor_members, "toarray"):
        n_oor_cells = oor_members.toarray().sum(axis=1)
    else:
        n_oor_cells = np.asarray(oor_members).sum(axis=1)

    total_cells_per_nhood = np.asarray(groups_mat.sum(axis=1)).ravel()
    frac_oor_cells = n_oor_cells / (total_cells_per_nhood + 1e-10)

    max_frac = float(frac_oor_cells.max()) if frac_oor_cells.size > 0 else 0.0
    oor_threshold = 0.2 * max_frac
    return (frac_oor_cells > oor_threshold).astype(int)



def _mapqc_runtime_config(mapqc_config: Mapping[str, Any] | None) -> Dict[str, Any]:
    mapqc_config = mapqc_config or {}
    return {
        "n_nhoods": mapqc_config.get("n_nhoods", 300),
        "k_min": mapqc_config.get("k_min", 1500),
        "k_max": mapqc_config.get("k_max", 10000),
        "study_key": mapqc_config.get("study_key", "dataset"),
        "seed": mapqc_config.get("seed", 10),
        "overwrite": mapqc_config.get("overwrite", True),
    }



def _collect_raw_scores(
    adata: AnnData,
    celltype_key: str,
    batch_key: str,
    mapqc_config: Mapping[str, Any] | None,
) -> Tuple[AnnData, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    target_celltype, query_mask = _prepare_case_control_labels(adata, celltype_key)

    raw_scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    print("→ Computing prob_uncertainty...")
    adata = prob_uncertainty(adata, label_key=celltype_key)
    y_true_query = (adata.obs.loc[query_mask, celltype_key] == target_celltype).astype(int).to_numpy()
    y_score_prob = _extract_query_scores(adata, "prob_uncertainty", query_mask)
    raw_scores["Prob_uncertainty"] = (_to_numpy(y_true_query), _to_numpy(y_score_prob))

    print("→ Computing dist_uncertainty...")
    adata = dist_uncertainty(adata, label_key=celltype_key)
    y_score_dist = _extract_query_scores(adata, "dist_uncertainty", query_mask)
    raw_scores["Dist_uncertainty"] = (_to_numpy(y_true_query), _to_numpy(y_score_dist))

    print("→ Computing DAlogFC...")
    adata = DALogFC(adata, celltype_key=celltype_key, batch_key=batch_key)
    y_true_da = _compute_dalogfc_targets(adata, celltype_key, target_celltype)
    y_score_da = _to_numpy(adata.uns["nhood_adata"].obs["logFC"])
    raw_scores["DAlogFC"] = (_to_numpy(y_true_da), _to_numpy(y_score_da))

    print("→ Computing mapQC...")
    cfg = _mapqc_runtime_config(mapqc_config)
    MapQC(
        adata,
        embedding="X",
        ref_query_key="ref_query",
        ref_key="ref",
        query_key="query",
        study_key=cfg["study_key"],
        batch_key=batch_key,
        n_nhoods=cfg["n_nhoods"],
        k_min=cfg["k_min"],
        k_max=cfg["k_max"],
        seed=cfg["seed"],
        overwrite=cfg["overwrite"],
        return_nhood_info_df=False,
        return_sample_dists_to_ref_df=False,
    )

    mapqc.evaluate(
        adata,
        case_control_key="case_control",
        case_cats=["case"],
        control_cats=["control"],
    )

    mapqc_valid = adata.obs["mapqc_score"].notna()
    y_true_mapqc = (adata.obs.loc[mapqc_valid, celltype_key] == target_celltype).astype(int).to_numpy()
    y_score_mapqc = adata.obs.loc[mapqc_valid, "mapqc_score"].to_numpy()
    raw_scores["mapQC"] = (_to_numpy(y_true_mapqc), _to_numpy(y_score_mapqc))

    return adata, raw_scores



def _compute_method_metrics(
    raw_scores: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    target_fdr: float,
) -> Dict[str, Dict[str, float]]:
    metrics_by_method: Dict[str, Dict[str, float]] = {}

    for method, (y_true, y_score) in raw_scores.items():
        y_true = _to_numpy(y_true)
        y_score = _to_numpy(y_score)
        metrics_by_method[method] = compute_identification_metrics(
            y_true=y_true,
            y_score=y_score,
            target_fdr=target_fdr,
        )

    return metrics_by_method



def run_evaluation(
    adata_path: AnnDataInput,
    celltype_key: str,
    batch_key: str,
    mapqc_config: Mapping[str, Any] | None,
    output_dir: Union[str, Path] = ".",
    target_fdr: float = 0.1,
) -> AnnData:
    """Run full mapAD evaluation and append results to ``evaluation_result.csv``.

    Notes
    -----
    The output file is tab-separated and contains:
    - legacy AUPRC columns: ``Prob_uncertainty``, ``Dist_uncertainty``, ``DAlogFC``, ``mapQC``
    - full prefixed metric columns, e.g. ``Prob_uncertainty_AUROC``
    """
    adata, adata_name = _load_adata(adata_path)
    _validate_adata_inputs(adata, celltype_key)

    adata, raw_scores = _collect_raw_scores(
        adata=adata,
        celltype_key=celltype_key,
        batch_key=batch_key,
        mapqc_config=mapqc_config,
    )
    metrics_by_method = _compute_method_metrics(raw_scores, target_fdr=target_fdr)

    row: MutableMapping[str, Any] = {"adata_name": adata_name}
    for method in METHOD_ORDER:
        method_metrics = metrics_by_method[method]
        row[method] = method_metrics["AUPRC"]
        for metric_name, metric_value in method_metrics.items():
            row[f"{method}_{metric_name}"] = metric_value

    output_path = Path(output_dir) / "evaluation_result.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([row])
    if output_path.exists():
        row_df.to_csv(output_path, index=False, mode="a", header=False, sep="\t")
    else:
        row_df.to_csv(output_path, index=False, sep="\t")

    return adata



def run_evaluation_for_auprc(
    adata_path: AnnDataInput,
    celltype_key: str,
    batch_key: str,
    mapqc_config: Mapping[str, Any] | None,
    output_dir: Union[str, Path] = ".",
    target_fdr: float = 0.1,
) -> Tuple[float, float, float, float]:
    """Run evaluation and return AUPRC values for all methods.

    Return order:
    ``(Prob_uncertainty, Dist_uncertainty, DAlogFC, mapQC)``
    """
    adata, _ = _load_adata(adata_path)
    _validate_adata_inputs(adata, celltype_key)

    _, raw_scores = _collect_raw_scores(
        adata=adata,
        celltype_key=celltype_key,
        batch_key=batch_key,
        mapqc_config=mapqc_config,
    )
    metrics_by_method = _compute_method_metrics(raw_scores, target_fdr=target_fdr)

    return tuple(float(metrics_by_method[method]["AUPRC"]) for method in METHOD_ORDER)



def run_prediction_step(
    adata_path: AnnDataInput,
    celltype_key: str,
    batch_key: str,
    mapqc_config: Mapping[str, Any] | None,
    output_file: Union[str, Path],
) -> Path:
    """Run heavy model scoring once and save raw scores for later metric computation.

    Output format (CSV):
    - method
    - y_true
    - y_score
    """
    adata, _ = _load_adata(adata_path)
    _validate_adata_inputs(adata, celltype_key)

    _, raw_scores = _collect_raw_scores(
        adata=adata,
        celltype_key=celltype_key,
        batch_key=batch_key,
        mapqc_config=mapqc_config,
    )

    raw_tables = []
    for method, (y_true, y_score) in raw_scores.items():
        raw_tables.append(
            pd.DataFrame(
                {
                    "method": method,
                    "y_true": _to_numpy(y_true).astype(int),
                    "y_score": _to_numpy(y_score),
                }
            )
        )

    score_file = Path(output_file)
    score_file.parent.mkdir(parents=True, exist_ok=True)

    pd.concat(raw_tables, axis=0, ignore_index=True).to_csv(score_file, index=False)
    return score_file



def compute_metrics_from_file(
    score_file: Union[str, Path],
    target_fdr: float = 0.1,
) -> Dict[str, float]:
    """Load raw scores and compute all identification metrics by method."""
    df = pd.read_csv(score_file)

    required_columns = {"method", "y_true", "y_score"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"score_file must contain columns {sorted(required_columns)}, got {list(df.columns)}"
        )

    method_metrics: Dict[str, float] = {}
    for method, group in df.groupby("method"):
        metrics = compute_identification_metrics(
            group["y_true"].to_numpy(),
            group["y_score"].to_numpy(),
            target_fdr=target_fdr,
        )
        for metric_name, metric_value in metrics.items():
            method_metrics[f"{method}_{metric_name}"] = float(metric_value)

    return method_metrics
