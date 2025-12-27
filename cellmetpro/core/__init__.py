"""Core module for metabolic profiling computations.

This module contains the fundamental algorithms and utilities for:
- COMPASS algorithm implementation for metabolic scoring
- Flux Balance Analysis (FBA) utilities
- Data preprocessing and normalization
- Microclustering for efficient computation
- Caching for optimization results
"""

from .compass import (
    CompassConfig,
    CompassResult,
    CompassScorer,
    run_compass,
)
from .fba import (
    FluxBalanceAnalyzer,
    apply_media,
    compute_yield,
    find_blocked_reactions,
    find_essential_reactions,
)
from .preprocessing import (
    DataLoader,
    filter_cells,
    filter_genes,
    normalize_expression,
    to_dataframe,
)
from .microclustering import (
    MicroclusterConfig,
    MicroclusterResult,
    microcluster,
    unpool_results,
)
from .cache import (
    CompassCache,
    MemoryCache,
    get_or_compute_max_fluxes,
)

__all__ = [
    # COMPASS
    "CompassConfig",
    "CompassResult",
    "CompassScorer",
    "run_compass",
    # FBA
    "FluxBalanceAnalyzer",
    "apply_media",
    "compute_yield",
    "find_blocked_reactions",
    "find_essential_reactions",
    # Preprocessing
    "DataLoader",
    "filter_cells",
    "filter_genes",
    "normalize_expression",
    "to_dataframe",
    # Microclustering
    "MicroclusterConfig",
    "MicroclusterResult",
    "microcluster",
    "unpool_results",
    # Cache
    "CompassCache",
    "MemoryCache",
    "get_or_compute_max_fluxes",
]
