"""Caching utilities for COMPASS optimization results.

This module provides caching functionality to store and retrieve
precomputed optimization results, avoiding redundant computation
when analyzing multiple samples with the same model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cellmetpro" / "cache"


class CompassCache:
    """Cache for COMPASS optimization results.

    Stores precomputed maximum fluxes and optimization results to
    speed up repeated analyses with the same model.

    Parameters
    ----------
    cache_dir : Path or str, optional
        Directory for cache files. Defaults to ~/.cellmetpro/cache.
    model_id : str, optional
        Identifier for the metabolic model. If None, computed from model.

    Attributes
    ----------
    cache_dir : Path
        Cache directory path.
    model_id : str
        Model identifier used for cache keys.

    Examples
    --------
    >>> from cellmetpro.core.cache import CompassCache
    >>> cache = CompassCache(model_id="human_gem_v1")
    >>> cache.save_max_fluxes(max_flux_dict)
    >>> max_fluxes = cache.load_max_fluxes()
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_id = model_id or "default"
        self._model_cache_dir = self.cache_dir / self.model_id
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory_cache: dict[str, Any] = {}

    @classmethod
    def from_model(
        cls,
        model: Any,
        cache_dir: Path | str | None = None,
    ) -> CompassCache:
        """Create cache with model-derived identifier.

        Parameters
        ----------
        model : cobra.Model
            Metabolic model to derive ID from.
        cache_dir : Path or str, optional
            Cache directory.

        Returns
        -------
        CompassCache
            Cache instance with model-specific ID.
        """
        # Create hash from model properties
        model_str = f"{model.id}_{len(model.reactions)}_{len(model.metabolites)}"
        model_hash = hashlib.md5(model_str.encode()).hexdigest()[:12]
        model_id = f"{model.id}_{model_hash}"

        return cls(cache_dir=cache_dir, model_id=model_id)

    def _get_path(self, key: str, suffix: str = ".pkl") -> Path:
        """Get file path for a cache key."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._model_cache_dir / f"{safe_key}{suffix}"

    def save_max_fluxes(self, max_fluxes: dict[str, float]) -> None:
        """Save precomputed maximum fluxes.

        Parameters
        ----------
        max_fluxes : dict[str, float]
            Dictionary mapping reaction IDs to maximum flux values.
        """
        path = self._get_path("max_fluxes")
        with open(path, "wb") as f:
            pickle.dump(max_fluxes, f)
        logger.debug(f"Saved max fluxes cache to {path}")

    def load_max_fluxes(self) -> dict[str, float] | None:
        """Load precomputed maximum fluxes.

        Returns
        -------
        dict[str, float] or None
            Maximum fluxes if cached, None otherwise.
        """
        path = self._get_path("max_fluxes")
        if path.exists():
            with open(path, "rb") as f:
                logger.debug(f"Loading max fluxes from cache: {path}")
                return pickle.load(f)
        return None

    def save_reaction_scores(
        self,
        scores: pd.DataFrame,
        sample_id: str,
    ) -> None:
        """Save reaction scores for a sample.

        Parameters
        ----------
        scores : pd.DataFrame
            Reaction scores (reactions x 1 or reactions as index).
        sample_id : str
            Sample identifier.
        """
        path = self._get_path(f"scores_{sample_id}", suffix=".parquet")
        scores.to_parquet(path)
        logger.debug(f"Saved scores for {sample_id}")

    def load_reaction_scores(self, sample_id: str) -> pd.DataFrame | None:
        """Load cached reaction scores for a sample.

        Parameters
        ----------
        sample_id : str
            Sample identifier.

        Returns
        -------
        pd.DataFrame or None
            Scores if cached, None otherwise.
        """
        path = self._get_path(f"scores_{sample_id}", suffix=".parquet")
        if path.exists():
            logger.debug(f"Loading scores for {sample_id} from cache")
            return pd.read_parquet(path)
        return None

    def save_penalties(
        self,
        penalties: pd.DataFrame,
        sample_id: str,
    ) -> None:
        """Save reaction penalties for a sample.

        Parameters
        ----------
        penalties : pd.DataFrame
            Penalty values.
        sample_id : str
            Sample identifier.
        """
        path = self._get_path(f"penalties_{sample_id}", suffix=".parquet")
        penalties.to_parquet(path)

    def load_penalties(self, sample_id: str) -> pd.DataFrame | None:
        """Load cached penalties for a sample.

        Parameters
        ----------
        sample_id : str
            Sample identifier.

        Returns
        -------
        pd.DataFrame or None
            Penalties if cached, None otherwise.
        """
        path = self._get_path(f"penalties_{sample_id}", suffix=".parquet")
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_json(self, data: dict, key: str) -> None:
        """Save arbitrary JSON-serializable data.

        Parameters
        ----------
        data : dict
            Data to save.
        key : str
            Cache key.
        """
        path = self._get_path(key, suffix=".json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, key: str) -> dict | None:
        """Load JSON data from cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        dict or None
            Data if cached, None otherwise.
        """
        path = self._get_path(key, suffix=".json")
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def has_sample(self, sample_id: str) -> bool:
        """Check if a sample has cached scores.

        Parameters
        ----------
        sample_id : str
            Sample identifier.

        Returns
        -------
        bool
            True if sample scores are cached.
        """
        path = self._get_path(f"scores_{sample_id}", suffix=".parquet")
        return path.exists()

    def get_cached_samples(self) -> list[str]:
        """Get list of samples with cached scores.

        Returns
        -------
        list[str]
            Sample IDs with cached data.
        """
        samples = []
        for path in self._model_cache_dir.glob("scores_*.parquet"):
            sample_id = path.stem.replace("scores_", "")
            samples.append(sample_id)
        return samples

    def clear(self, sample_id: str | None = None) -> None:
        """Clear cache data.

        Parameters
        ----------
        sample_id : str, optional
            If provided, clear only this sample. Otherwise, clear all.
        """
        if sample_id is not None:
            # Clear specific sample
            for suffix in [".parquet", ".pkl"]:
                for prefix in ["scores_", "penalties_"]:
                    path = self._get_path(f"{prefix}{sample_id}", suffix=suffix)
                    if path.exists():
                        path.unlink()
            logger.info(f"Cleared cache for sample: {sample_id}")
        else:
            # Clear all
            shutil.rmtree(self._model_cache_dir)
            self._model_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared all cache for model: {self.model_id}")

    def get_cache_size(self) -> int:
        """Get total cache size in bytes.

        Returns
        -------
        int
            Total size of cached files in bytes.
        """
        total = 0
        for path in self._model_cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def get_cache_info(self) -> dict:
        """Get cache information.

        Returns
        -------
        dict
            Cache statistics and info.
        """
        samples = self.get_cached_samples()
        size = self.get_cache_size()

        return {
            "model_id": self.model_id,
            "cache_dir": str(self._model_cache_dir),
            "n_cached_samples": len(samples),
            "cached_samples": samples,
            "total_size_bytes": size,
            "total_size_mb": size / (1024 * 1024),
            "has_max_fluxes": self._get_path("max_fluxes").exists(),
        }


def get_or_compute_max_fluxes(
    model: Any,
    cache: CompassCache | None = None,
) -> dict[str, float]:
    """Get maximum fluxes from cache or compute them.

    Parameters
    ----------
    model : cobra.Model
        Metabolic model.
    cache : CompassCache, optional
        Cache instance to use.

    Returns
    -------
    dict[str, float]
        Maximum flux for each reaction.
    """
    import cobra

    # Try loading from cache
    if cache is not None:
        cached = cache.load_max_fluxes()
        if cached is not None:
            logger.info(f"Loaded {len(cached)} max fluxes from cache")
            return cached

    logger.info("Computing maximum fluxes...")
    max_fluxes = {}

    for rxn in model.reactions:
        if rxn.boundary:
            continue

        with model:
            model.objective = rxn
            model.objective_direction = "max"

            try:
                solution = model.optimize()
                if solution.status == "optimal":
                    max_fluxes[rxn.id] = abs(solution.objective_value)
                else:
                    max_fluxes[rxn.id] = 0.0
            except Exception:
                max_fluxes[rxn.id] = 0.0

    # Save to cache
    if cache is not None:
        cache.save_max_fluxes(max_fluxes)
        logger.info(f"Cached {len(max_fluxes)} max fluxes")

    return max_fluxes


class MemoryCache:
    """Simple in-memory cache for temporary results.

    Useful for caching intermediate results within a single session.

    Parameters
    ----------
    max_size : int, default=1000
        Maximum number of items to cache.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []
        self.max_size = max_size

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.

        Parameters
        ----------
        key : str
            Cache key.
        default : Any, optional
            Default value if key not found.

        Returns
        -------
        Any
            Cached value or default.
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return default

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.
        """
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict least recently used
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = value
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._access_order.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)
