"""Models module for genome-scale metabolic models.

This module handles loading and managing GEM (Genome-Scale Metabolic)
models for different organisms. Supports SBML, JSON, and MAT formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cobra

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Path to bundled GEM models
GEMS_DIR = Path(__file__).parent / "gems"

# Supported model formats
SUPPORTED_FORMATS = {".xml", ".sbml", ".json", ".mat", ".yml", ".yaml"}

# URLs for downloading standard models
MODEL_URLS = {
    "human": "https://github.com/SysBioChalmers/Human-GEM/raw/main/model/Human-GEM.xml",
    "mouse": "https://github.com/SysBioChalmers/Mouse-GEM/raw/main/model/Mouse-GEM.xml",
    "recon2": "https://www.vmh.life/files/reconstructions/ReconMaps/Recon2.v04.mat",
    "recon3d": "https://www.vmh.life/files/reconstructions/ReconMaps/Recon3D.mat",
}


def load_human_gem() -> cobra.Model:
    """Load the human genome-scale metabolic model.

    Attempts to load from bundled models first, then downloads if needed.

    Returns
    -------
    cobra.Model
        Human GEM model.

    Raises
    ------
    FileNotFoundError
        If model cannot be found or downloaded.
    """
    return load_gem("human")


def load_mouse_gem() -> cobra.Model:
    """Load the mouse genome-scale metabolic model.

    Returns
    -------
    cobra.Model
        Mouse GEM model.

    Raises
    ------
    FileNotFoundError
        If model cannot be found or downloaded.
    """
    return load_gem("mouse")


def load_gem(organism: str) -> cobra.Model:
    """Load a GEM model by organism name or file path.

    Parameters
    ----------
    organism : str
        Organism name ('human', 'mouse', 'recon2', 'recon3d') or path to model file.

    Returns
    -------
    cobra.Model
        The loaded metabolic model.

    Raises
    ------
    ValueError
        If organism is not supported and path doesn't exist.
    FileNotFoundError
        If model file cannot be found.
    """
    # Check if it's a file path
    path = Path(organism)
    if path.exists():
        return load_model_from_file(path)

    # Check bundled models directory
    organism_lower = organism.lower()
    for suffix in SUPPORTED_FORMATS:
        bundled_path = GEMS_DIR / f"{organism_lower}{suffix}"
        if bundled_path.exists():
            logger.info(f"Loading bundled model from {bundled_path}")
            return load_model_from_file(bundled_path)

    # Check for common model names
    if organism_lower in MODEL_URLS:
        # Try to find a cached version first
        cache_dir = GEMS_DIR / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for suffix in [".xml", ".json", ".mat"]:
            cached_path = cache_dir / f"{organism_lower}{suffix}"
            if cached_path.exists():
                logger.info(f"Loading cached model from {cached_path}")
                return load_model_from_file(cached_path)

        raise FileNotFoundError(
            f"Model '{organism}' not found. Please download it manually from:\n"
            f"  {MODEL_URLS[organism_lower]}\n"
            f"and place it in: {GEMS_DIR}"
        )

    raise ValueError(
        f"Unknown organism '{organism}'. Supported organisms: {list(MODEL_URLS.keys())}\n"
        f"Or provide a path to a model file ({', '.join(SUPPORTED_FORMATS)})"
    )


def load_model_from_file(path: str | Path) -> cobra.Model:
    """Load a metabolic model from a file.

    Parameters
    ----------
    path : str or Path
        Path to the model file. Supports SBML (.xml, .sbml), JSON (.json),
        MATLAB (.mat), and YAML (.yml, .yaml) formats.

    Returns
    -------
    cobra.Model
        The loaded COBRApy model.

    Raises
    ------
    ValueError
        If the file format is not supported.
    FileNotFoundError
        If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    suffix = path.suffix.lower()
    logger.info(f"Loading model from {path} (format: {suffix})")

    if suffix in {".xml", ".sbml"}:
        model = cobra.io.read_sbml_model(str(path))
    elif suffix == ".json":
        model = cobra.io.load_json_model(str(path))
    elif suffix == ".mat":
        model = cobra.io.load_matlab_model(str(path))
    elif suffix in {".yml", ".yaml"}:
        model = cobra.io.load_yaml_model(str(path))
    else:
        raise ValueError(
            f"Unsupported model format: {suffix}. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    logger.info(
        f"Loaded model '{model.id}' with {len(model.reactions)} reactions, "
        f"{len(model.metabolites)} metabolites, {len(model.genes)} genes"
    )
    return model


def prepare_model_for_compass(
    model: cobra.Model,
    exchange_limit: float = 1000.0,
    make_irreversible: bool = True,
) -> cobra.Model:
    """Prepare a metabolic model for COMPASS analysis.

    This function prepares the model by:
    1. Limiting exchange reaction bounds
    2. Optionally converting to irreversible format

    Parameters
    ----------
    model : cobra.Model
        The input COBRApy model.
    exchange_limit : float, default=1000.0
        Maximum absolute flux for exchange reactions.
    make_irreversible : bool, default=True
        If True, split reversible reactions into forward/reverse pairs.

    Returns
    -------
    cobra.Model
        Prepared model (copy of original).
    """
    # Work on a copy
    model = model.copy()

    # Limit exchange reactions
    for rxn in model.exchanges:
        if rxn.lower_bound < -exchange_limit:
            rxn.lower_bound = -exchange_limit
        if rxn.upper_bound > exchange_limit:
            rxn.upper_bound = exchange_limit

    # Convert to irreversible if requested
    if make_irreversible:
        model = _make_irreversible(model)

    return model


def _make_irreversible(model: cobra.Model) -> cobra.Model:
    """Convert reversible reactions to irreversible pairs.

    For each reversible reaction, creates a forward and reverse reaction
    with non-negative bounds.

    Parameters
    ----------
    model : cobra.Model
        Input model.

    Returns
    -------
    cobra.Model
        Model with only irreversible reactions.
    """
    # Identify reversible reactions
    reversible_rxns = [
        rxn for rxn in model.reactions if rxn.lower_bound < 0 and rxn.upper_bound > 0
    ]

    for rxn in reversible_rxns:
        # Create reverse reaction
        reverse_rxn = cobra.Reaction(
            id=f"{rxn.id}_reverse",
            name=f"{rxn.name} (reverse)" if rxn.name else None,
            lower_bound=0,
            upper_bound=-rxn.lower_bound,
        )

        # Add reverse stoichiometry
        reverse_rxn.add_metabolites({met: -coef for met, coef in rxn.metabolites.items()})

        # Copy gene reaction rule
        reverse_rxn.gene_reaction_rule = rxn.gene_reaction_rule

        # Update original reaction to forward only
        rxn.lower_bound = 0

        # Add reverse reaction to model
        model.add_reactions([reverse_rxn])

    return model


def get_reaction_gene_mapping(model: cobra.Model) -> dict[str, list[str]]:
    """Get mapping from reactions to their associated genes.

    Parameters
    ----------
    model : cobra.Model
        A COBRApy model.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping reaction IDs to lists of gene IDs.
    """
    mapping = {}
    for rxn in model.reactions:
        gene_ids = [g.id for g in rxn.genes]
        if gene_ids:
            mapping[rxn.id] = gene_ids
    return mapping


def get_subsystem_reactions(model: cobra.Model) -> dict[str, list[str]]:
    """Get mapping from subsystems to their reactions.

    Parameters
    ----------
    model : cobra.Model
        A COBRApy model.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping subsystem names to lists of reaction IDs.
    """
    subsystems: dict[str, list[str]] = {}
    for rxn in model.reactions:
        if rxn.subsystem:
            if rxn.subsystem not in subsystems:
                subsystems[rxn.subsystem] = []
            subsystems[rxn.subsystem].append(rxn.id)
    return subsystems
