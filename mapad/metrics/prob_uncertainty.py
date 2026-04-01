from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.neighbors import KNeighborsTransformer


def _get_embedding(adata: AnnData, embedding: str):
    if embedding == "X":
        return adata.X
    if embedding in adata.obsm:
        return adata.obsm[embedding]
    raise KeyError(f"Embedding '{embedding}' not found in adata.obsm.")


def prob_uncertainty(
    adata: AnnData,
    embedding: str = "X",
    label_key: str = "cell_type",
    ref_query_key: str = "ref_query",
    ref_key: str = "ref",
    query_key: str = "query",
    n_neighbors: int = 50,
    threshold: float = 1.0,
    pred_unknown: bool = False,
    return_adata: bool = True,
) -> Union[AnnData, pd.Series]:
    """Estimate uncertainty from weighted KNN label probabilities.

    For each query cell, uncertainty is defined as ``1 - max_class_probability``
    where class probabilities are computed from distance-weighted KNN votes.
    """
    ref_adata = adata[adata.obs[ref_query_key] == ref_key]
    query_adata = adata[adata.obs[ref_query_key] == query_key]

    if ref_adata.n_obs == 0 or query_adata.n_obs == 0:
        empty = pd.Series(dtype=float, name="prob_uncertainty")
        adata.uns["prob_uncertainty"] = empty
        return adata if return_adata else empty

    ref_emb = _get_embedding(ref_adata, embedding)
    query_emb = _get_embedding(query_adata, embedding)

    y_train_labels = ref_adata.obs[label_key].to_numpy()
    n_neighbors = max(1, min(n_neighbors, ref_adata.n_obs))

    knn = KNeighborsTransformer(
        n_neighbors=n_neighbors,
        mode="distance",
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(ref_emb)

    top_k_distances, top_k_indices = knn.kneighbors(query_emb)

    # Distance-adaptive scaling, protected against zero variance.
    stds = np.std(top_k_distances, axis=1)
    stds = np.maximum(stds, 1e-12)
    scales = ((2.0 / stds) ** 2).reshape(-1, 1)

    weights = np.exp(-top_k_distances / scales)
    weights /= np.sum(weights, axis=1, keepdims=True)

    pred_labels = []
    uncertainties = []

    for i in range(query_adata.n_obs):
        neighbor_labels = y_train_labels[top_k_indices[i]]
        unique_labels = np.unique(neighbor_labels)

        best_label = None
        best_prob = 0.0
        for candidate_label in unique_labels:
            candidate_prob = weights[i, neighbor_labels == candidate_label].sum()
            if candidate_prob > best_prob:
                best_prob = candidate_prob
                best_label = candidate_label

        pred_labels.append("Unknown" if pred_unknown and best_prob < threshold else best_label)
        uncertainties.append(max(1 - best_prob, 0.0))

    pred_labels = pd.Series(pred_labels, index=query_adata.obs_names, name="prob_pred_label")
    uncertainties = pd.Series(uncertainties, index=query_adata.obs_names, name="prob_uncertainty")

    adata.obs["prob_pred_label"] = "Reference"
    adata.obs.loc[query_adata.obs_names, "prob_pred_label"] = pred_labels
    adata.uns["prob_uncertainty"] = uncertainties

    return adata if return_adata else uncertainties
