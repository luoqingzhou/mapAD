from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def _get_embedding(adata: AnnData, embedding: str):
    if embedding == "X":
        return adata.X
    if embedding in adata.obsm:
        return adata.obsm[embedding]
    raise KeyError(f"Embedding '{embedding}' not found in adata.obsm.")


def _to_dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


def dist_uncertainty(
    adata: AnnData,
    embedding: str = "X",
    label_key: str = "cell_type",
    ref_query_key: str = "ref_query",
    ref_key: str = "ref",
    query_key: str = "query",
    scale: bool = True,
    return_adata: bool = True,
) -> Union[AnnData, pd.Series]:
    """Estimate uncertainty by distance to closest reference prototype.

    For each query cell, the uncertainty score is the minimum Euclidean distance
    to label-wise centroids computed from the reference cells.
    """
    ref_mask = adata.obs[ref_query_key] == ref_key
    query_mask = adata.obs[ref_query_key] == query_key

    ref_adata = adata[ref_mask]
    query_adata = adata[query_mask]

    if ref_adata.n_obs == 0 or query_adata.n_obs == 0:
        empty = pd.Series(dtype=float, name="dist_uncertainty")
        adata.uns["dist_uncertainty"] = empty
        return adata if return_adata else empty

    ref_emb = _to_dense(_get_embedding(ref_adata, embedding))
    query_emb = _to_dense(_get_embedding(query_adata, embedding))

    labels = ref_adata.obs[label_key]
    unique_labels = labels.unique()

    prototypes = []
    for label in unique_labels:
        label_mask = (labels == label).to_numpy()
        prototypes.append(ref_emb[label_mask].mean(axis=0))
    prototypes = np.stack(prototypes)

    distances = cdist(query_emb, prototypes, metric="euclidean")
    min_dist = np.min(distances, axis=1)
    best_proto_idx = np.argmin(distances, axis=1)
    pred_labels = unique_labels[best_proto_idx]

    if scale:
        scaled = RobustScaler().fit_transform(min_dist.reshape(-1, 1))
        uncertainties = MinMaxScaler(feature_range=(0, 1)).fit_transform(scaled).flatten()
    else:
        uncertainties = min_dist

    res_uncertainty = pd.Series(uncertainties, index=query_adata.obs_names, name="dist_uncertainty")
    res_preds = pd.Series(pred_labels, index=query_adata.obs_names, name="dist_pred_label")

    adata.obs["dist_uncertainty"] = np.nan
    adata.obs.loc[query_adata.obs_names, "dist_uncertainty"] = res_uncertainty

    adata.obs["dist_pred_label"] = "Reference"
    adata.obs.loc[query_adata.obs_names, "dist_pred_label"] = res_preds

    adata.uns["dist_uncertainty"] = res_uncertainty

    return adata if return_adata else res_uncertainty
