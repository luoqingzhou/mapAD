from __future__ import annotations

import milopy
import pandas as pd
import scanpy as sc
from anndata import AnnData

# Compatibility patch for packages expecting pandas<2 ``iteritems``.
pd.DataFrame.iteritems = pd.DataFrame.items


def run_milo(
    adata: AnnData,
    ref_query_key: str = "ref_query",
    ref_key: str = "ref",
    query_key: str = "query",
    batch_key: str = "sample_id",
    celltype_key: str = "cell_type",
    design: str = "~is_query",
) -> None:
    """Run Milo neighborhood construction and differential abundance test."""
    milopy.core.make_nhoods(adata, prop=0.1)
    milopy.core.count_nhoods(adata, sample_col=batch_key)
    milopy.utils.annotate_nhoods(adata[adata.obs[ref_query_key] == ref_key], celltype_key)
    adata.obs["is_query"] = adata.obs[ref_query_key] == query_key
    milopy.core.DA_nhoods(adata, design=design)


def DALogFC(
    adata: AnnData,
    embedding: str = "X",
    ref_query_key: str = "ref_query",
    ref_key: str = "ref",
    query_key: str = "query",
    batch_key: str = "sample_id",
    celltype_key: str = "cell_type",
    milo_design: str = "~is_query",
    **kwargs,
) -> AnnData:
    """Compute DAlogFC-based OOR scores and store Milo neighborhood outputs.

    Results are written into:
    - ``adata.uns['nhood_adata']`` (from Milo)
    - ``adata.uns['sample_adata']`` (transposed neighborhood view with OOR metadata)
    """
    n_controls = adata[adata.obs[ref_query_key] == ref_key].obs[batch_key].nunique()
    n_queries = adata[adata.obs[ref_query_key] == query_key].obs[batch_key].nunique()

    # Cap k to avoid memory issues on large datasets.
    k_neighbors = min((n_controls + n_queries) * 5, 200)
    k_neighbors = max(k_neighbors, 1)

    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=embedding, n_neighbors=k_neighbors)

    run_milo(
        adata,
        ref_query_key=ref_query_key,
        ref_key=ref_key,
        query_key=query_key,
        batch_key=batch_key,
        celltype_key=celltype_key,
        design=milo_design,
    )

    sample_adata = adata.uns["nhood_adata"].T.copy()
    sample_adata.var["OOR_score"] = sample_adata.var["logFC"].copy()
    sample_adata.var["OOR_signif"] = (
        (sample_adata.var["SpatialFDR"] < 0.1) & (sample_adata.var["logFC"] > 0)
    ).astype(int)
    sample_adata.varm["groups"] = adata.obsm["nhoods"].T
    adata.uns["sample_adata"] = sample_adata
    return adata
