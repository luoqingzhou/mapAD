from __future__ import annotations

from anndata import AnnData
from mapqc import run_mapqc


def MapQC(
    adata: AnnData,
    embedding: str = "X",
    ref_query_key: str = "ref_query",
    ref_key: str = "ref",
    query_key: str = "query",
    batch_key: str = "sample_id",
    study_key: str = "dataset_id",
    **kwargs,
) -> AnnData:
    """Run MapQC and attach results to ``adata``.

    Parameters
    ----------
    adata:
        Combined reference + query AnnData object.
    embedding:
        Embedding key used by MapQC ("X" or an ``.obsm`` key).
    ref_query_key:
        Column in ``adata.obs`` distinguishing reference/query cells.
    ref_key:
        Category value identifying reference cells.
    query_key:
        Category value identifying query cells.
    batch_key:
        Sample/batch column used by MapQC.
    study_key:
        Study/source column used by MapQC.
    kwargs:
        Extra keyword arguments forwarded to ``mapqc.run_mapqc``.
    """
    return run_mapqc(
        adata=adata,
        adata_emb_loc=embedding,
        ref_q_key=ref_query_key,
        q_cat=query_key,
        r_cat=ref_key,
        sample_key=batch_key,
        study_key=study_key,
        **kwargs,
    )
