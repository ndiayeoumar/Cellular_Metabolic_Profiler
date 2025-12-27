"""Microclustering for efficient COMPASS computation.

Microclustering groups similar cells together and computes COMPASS
scores on the aggregated clusters, then propagates results back to
individual cells. This significantly speeds up computation for large
datasets.

References:
    DeTomaso & Yosef (2021) "Hotspot identifies informative gene modules
    across modalities of single-cell genomics" Cell Systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

# Default seeds for reproducibility
PCA_SEED = 1337
LEIDEN_SEED = 1337
KMEANS_SEED = 1337


@dataclass
class MicroclusterConfig:
    """Configuration for microclustering.

    Attributes
    ----------
    cells_per_cluster : int
        Target number of cells per microcluster.
    n_neighbors : int
        Number of neighbors for KNN graph construction.
    n_pcs : int
        Number of principal components to use.
    resolution : float
        Resolution parameter for Leiden clustering.
    method : str
        Clustering method: 'leiden', 'kmeans', or 'knn'.
    min_cluster_size : int
        Minimum cells per cluster (smaller clusters are merged).
    random_state : int
        Random seed for reproducibility.
    """

    cells_per_cluster: int = 100
    n_neighbors: int = 30
    n_pcs: int = 20
    resolution: float = 1.0
    method: Literal["leiden", "kmeans", "knn"] = "leiden"
    min_cluster_size: int = 10
    random_state: int = LEIDEN_SEED


@dataclass
class MicroclusterResult:
    """Results from microclustering.

    Attributes
    ----------
    cluster_labels : np.ndarray
        Cluster assignment for each cell.
    n_clusters : int
        Number of microclusters.
    pooled_expression : pd.DataFrame
        Aggregated expression matrix (genes x clusters).
    cell_to_cluster : dict[str, int]
        Mapping from cell names to cluster indices.
    cluster_sizes : np.ndarray
        Number of cells in each cluster.
    """

    cluster_labels: np.ndarray
    n_clusters: int
    pooled_expression: pd.DataFrame
    cell_to_cluster: dict[str, int]
    cluster_sizes: np.ndarray


def microcluster(
    expression: pd.DataFrame | ad.AnnData,
    config: MicroclusterConfig | None = None,
    knn_indices: np.ndarray | None = None,
    knn_distances: np.ndarray | None = None,
) -> MicroclusterResult:
    """Perform microclustering on expression data.

    Groups similar cells into microclusters for efficient computation.
    Expression values are aggregated (mean) within each cluster.

    Parameters
    ----------
    expression : pd.DataFrame or AnnData
        Gene expression matrix. DataFrame: genes x cells. AnnData: cells x genes.
    config : MicroclusterConfig, optional
        Configuration parameters.
    knn_indices : np.ndarray, optional
        Pre-computed KNN indices (cells x k).
    knn_distances : np.ndarray, optional
        Pre-computed KNN distances (cells x k).

    Returns
    -------
    MicroclusterResult
        Microclustering results including cluster assignments and
        pooled expression matrix.

    Examples
    --------
    >>> from cellmetpro.core.microclustering import microcluster, MicroclusterConfig
    >>> config = MicroclusterConfig(cells_per_cluster=50, method='leiden')
    >>> result = microcluster(expression_data, config)
    >>> print(f"Created {result.n_clusters} microclusters")
    """
    config = config or MicroclusterConfig()

    # Convert expression to standard format
    expr_df, cell_names = _process_expression(expression)
    n_cells = len(cell_names)

    logger.info(f"Microclustering {n_cells} cells...")

    # Estimate target number of clusters
    target_n_clusters = max(1, n_cells // config.cells_per_cluster)
    logger.info(f"Target: ~{target_n_clusters} clusters")

    # Get expression matrix (cells x genes)
    expr_matrix = expr_df.T.values

    # Compute PCA if needed
    n_pcs = min(config.n_pcs, expr_matrix.shape[1], expr_matrix.shape[0] - 1)
    pca = PCA(n_components=n_pcs, random_state=config.random_state)
    pca_coords = pca.fit_transform(expr_matrix)

    # Compute KNN graph if not provided
    if knn_indices is None:
        n_neighbors = min(config.n_neighbors + 1, n_cells)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(pca_coords)
        knn_distances, knn_indices = knn.kneighbors(pca_coords)

    # Perform clustering
    if config.method == "leiden":
        labels = _leiden_clustering(
            pca_coords,
            knn_indices,
            knn_distances,
            config.resolution,
            config.random_state,
        )
    elif config.method == "kmeans":
        labels = _kmeans_clustering(
            pca_coords, target_n_clusters, config.random_state
        )
    elif config.method == "knn":
        labels = _knn_partitioning(
            knn_indices, target_n_clusters, config.random_state
        )
    else:
        raise ValueError(f"Unknown clustering method: {config.method}")

    # Adjust clusters to target size
    labels = _readjust_clusters(
        pca_coords,
        labels,
        target_n_clusters,
        config.min_cluster_size,
        config.random_state,
    )

    n_clusters = len(np.unique(labels))
    logger.info(f"Created {n_clusters} microclusters")

    # Pool expression by cluster
    pooled_expr = _pool_expression(expr_df, labels, cell_names)

    # Compute cluster sizes
    cluster_sizes = np.bincount(labels)

    # Create cell to cluster mapping
    cell_to_cluster = {cell_names[i]: labels[i] for i in range(n_cells)}

    return MicroclusterResult(
        cluster_labels=labels,
        n_clusters=n_clusters,
        pooled_expression=pooled_expr,
        cell_to_cluster=cell_to_cluster,
        cluster_sizes=cluster_sizes,
    )


def unpool_results(
    cluster_results: pd.DataFrame,
    microcluster_result: MicroclusterResult,
    cell_names: list[str] | None = None,
) -> pd.DataFrame:
    """Propagate cluster-level results back to individual cells.

    Parameters
    ----------
    cluster_results : pd.DataFrame
        Results computed on microclusters (features x clusters).
    microcluster_result : MicroclusterResult
        Microclustering results with cell-to-cluster mapping.
    cell_names : list[str], optional
        Cell names in desired order. If None, uses original order.

    Returns
    -------
    pd.DataFrame
        Results expanded to individual cells (features x cells).
    """
    if cell_names is None:
        cell_names = list(microcluster_result.cell_to_cluster.keys())

    # Map each cell to its cluster's results
    cell_results = {}
    for cell_name in cell_names:
        cluster_idx = microcluster_result.cell_to_cluster[cell_name]
        cluster_col = cluster_results.columns[cluster_idx]
        cell_results[cell_name] = cluster_results[cluster_col]

    return pd.DataFrame(cell_results)


def _process_expression(
    expression: pd.DataFrame | ad.AnnData,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert expression input to genes x cells DataFrame."""
    try:
        import anndata as ad

        if isinstance(expression, ad.AnnData):
            if issparse(expression.X):
                data = expression.X.toarray().T
            else:
                data = expression.X.T
            df = pd.DataFrame(
                data, index=expression.var_names, columns=expression.obs_names
            )
            return df, list(expression.obs_names)
    except ImportError:
        pass

    if isinstance(expression, pd.DataFrame):
        return expression, list(expression.columns)

    raise TypeError(f"Unsupported expression type: {type(expression)}")


def _leiden_clustering(
    coords: np.ndarray,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    resolution: float,
    random_state: int,
) -> np.ndarray:
    """Perform Leiden clustering on KNN graph."""
    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        logger.warning("leidenalg not installed, falling back to k-means")
        return _kmeans_clustering(coords, len(coords) // 100, random_state)

    n_cells = len(coords)

    # Build weighted adjacency from KNN
    # Apply Gaussian kernel to distances
    median_dist = np.median(knn_distances[:, 1:])
    sigma = median_dist if median_dist > 0 else 1.0

    edges = []
    weights = []

    for i in range(n_cells):
        for j_idx in range(1, knn_indices.shape[1]):  # Skip self
            j = knn_indices[i, j_idx]
            if j < n_cells:
                dist = knn_distances[i, j_idx]
                weight = np.exp(-(dist**2) / (2 * sigma**2))
                edges.append((i, j))
                weights.append(weight)

    # Create igraph graph
    g = ig.Graph(n=n_cells, edges=edges, directed=False)
    g.es["weight"] = weights

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )

    return np.array(partition.membership)


def _kmeans_clustering(
    coords: np.ndarray, n_clusters: int, random_state: int
) -> np.ndarray:
    """Perform k-means clustering."""
    from sklearn.cluster import KMeans

    n_clusters = max(1, min(n_clusters, len(coords)))

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    )
    labels = kmeans.fit_predict(coords)

    return labels


def _knn_partitioning(
    knn_indices: np.ndarray, n_clusters: int, random_state: int
) -> np.ndarray:
    """Partition cells based on KNN graph connectivity."""
    # Simple approach: use connected components with random seeds
    np.random.seed(random_state)

    n_cells = len(knn_indices)
    labels = np.full(n_cells, -1)

    # Select random seed cells
    seeds = np.random.choice(n_cells, min(n_clusters, n_cells), replace=False)

    for cluster_idx, seed in enumerate(seeds):
        labels[seed] = cluster_idx

    # Propagate labels through KNN graph
    for _ in range(10):  # Iterations
        for i in range(n_cells):
            if labels[i] == -1:
                # Assign to most common neighbor label
                neighbor_labels = labels[knn_indices[i]]
                valid_labels = neighbor_labels[neighbor_labels >= 0]
                if len(valid_labels) > 0:
                    labels[i] = np.bincount(valid_labels).argmax()

    # Assign remaining unlabeled cells
    for i in range(n_cells):
        if labels[i] == -1:
            labels[i] = 0

    return labels


def _readjust_clusters(
    coords: np.ndarray,
    labels: np.ndarray,
    target_n_clusters: int,
    min_cluster_size: int,
    random_state: int,
) -> np.ndarray:
    """Readjust cluster sizes to meet target.

    Splits large clusters and merges small ones.
    """
    labels = labels.copy()
    n_cells = len(labels)

    # Get cluster sizes
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # If too few clusters, split large ones
    while n_clusters < target_n_clusters:
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        largest_cluster = np.argmax(cluster_sizes)

        if cluster_sizes[largest_cluster] < 2 * min_cluster_size:
            break

        # Split largest cluster using k-means
        cluster_mask = labels == largest_cluster
        cluster_coords = coords[cluster_mask]

        if len(cluster_coords) >= 2:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=5)
            sub_labels = kmeans.fit_predict(cluster_coords)

            # Assign new labels
            new_label = labels.max() + 1
            cluster_indices = np.where(cluster_mask)[0]
            for i, idx in enumerate(cluster_indices):
                if sub_labels[i] == 1:
                    labels[idx] = new_label

        n_clusters = len(np.unique(labels))

    # Merge small clusters
    unique_labels = np.unique(labels)
    cluster_sizes = {l: np.sum(labels == l) for l in unique_labels}

    for label in unique_labels:
        if cluster_sizes[label] < min_cluster_size:
            # Find nearest large cluster
            cluster_mask = labels == label
            cluster_center = coords[cluster_mask].mean(axis=0)

            best_target = None
            best_dist = np.inf

            for other_label in unique_labels:
                if other_label != label and cluster_sizes[other_label] >= min_cluster_size:
                    other_mask = labels == other_label
                    other_center = coords[other_mask].mean(axis=0)
                    dist = np.linalg.norm(cluster_center - other_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_target = other_label

            if best_target is not None:
                labels[cluster_mask] = best_target
                cluster_sizes[best_target] += cluster_sizes[label]
                cluster_sizes[label] = 0

    # Relabel to consecutive integers
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    return labels


def _pool_expression(
    expression: pd.DataFrame,
    labels: np.ndarray,
    cell_names: list[str],
) -> pd.DataFrame:
    """Aggregate expression by cluster (mean).

    Parameters
    ----------
    expression : pd.DataFrame
        Gene expression (genes x cells).
    labels : np.ndarray
        Cluster labels for each cell.
    cell_names : list[str]
        Cell names matching expression columns.

    Returns
    -------
    pd.DataFrame
        Pooled expression (genes x clusters).
    """
    n_clusters = len(np.unique(labels))
    pooled = np.zeros((expression.shape[0], n_clusters))

    for cluster_idx in range(n_clusters):
        cluster_mask = labels == cluster_idx
        cluster_cells = [cell_names[i] for i in range(len(labels)) if cluster_mask[i]]
        if cluster_cells:
            pooled[:, cluster_idx] = expression[cluster_cells].mean(axis=1)

    cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
    return pd.DataFrame(pooled, index=expression.index, columns=cluster_names)


def filter_genes_fano(
    expression: pd.DataFrame,
    n_genes: int = 2000,
    min_mean: float = 0.01,
) -> pd.DataFrame:
    """Select highly variable genes using Fano factor.

    Parameters
    ----------
    expression : pd.DataFrame
        Gene expression (genes x cells).
    n_genes : int
        Number of genes to select.
    min_mean : float
        Minimum mean expression threshold.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix.
    """
    means = expression.mean(axis=1)
    variances = expression.var(axis=1)

    # Filter by minimum expression
    valid = means >= min_mean
    means = means[valid]
    variances = variances[valid]

    # Compute Fano factor (variance / mean)
    fano = variances / (means + 1e-10)

    # Select top genes
    top_genes = fano.nlargest(n_genes).index
    return expression.loc[top_genes]


def filter_genes_threshold(
    expression: pd.DataFrame,
    min_cells: int = 10,
    min_counts: float = 1.0,
) -> pd.DataFrame:
    """Filter genes by expression threshold.

    Parameters
    ----------
    expression : pd.DataFrame
        Gene expression (genes x cells).
    min_cells : int
        Minimum number of cells expressing the gene.
    min_counts : float
        Minimum expression value to count as expressed.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix.
    """
    n_expressing = (expression >= min_counts).sum(axis=1)
    valid_genes = n_expressing >= min_cells
    return expression.loc[valid_genes]
