"""Flux Balance Analysis (FBA) utilities.

This module provides utilities for running FBA on genome-scale
metabolic models, including constraint-based optimization and
flux variability analysis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

import cobra

logger = logging.getLogger(__name__)


class FluxBalanceAnalyzer:
    """Perform Flux Balance Analysis on metabolic models.

    This class provides methods for running FBA, flux variability
    analysis (FVA), and manipulating model constraints.

    Parameters
    ----------
    model : cobra.Model
        A genome-scale metabolic model.

    Attributes
    ----------
    solution : cobra.Solution
        The FBA solution after optimization.

    Examples
    --------
    >>> from cellmetpro.core.fba import FluxBalanceAnalyzer
    >>> fba = FluxBalanceAnalyzer(model)
    >>> fluxes = fba.optimize()
    >>> print(f"Optimal objective: {fba.solution.objective_value}")
    """

    def __init__(self, model: cobra.Model) -> None:
        self.model = model.copy()
        self.solution: cobra.Solution | None = None
        self._original_bounds: dict[str, tuple[float, float]] = {}

    def optimize(
        self,
        objective: str | None = None,
        direction: str = "max",
    ) -> pd.Series:
        """Run FBA optimization.

        Parameters
        ----------
        objective : str, optional
            Reaction ID to optimize. If None, uses model default.
        direction : str, default="max"
            Optimization direction: "max" or "min".

        Returns
        -------
        pd.Series
            Flux values for all reactions.

        Raises
        ------
        cobra.exceptions.OptimizationError
            If optimization fails.
        """
        if objective is not None:
            rxn = self.model.reactions.get_by_id(objective)
            self.model.objective = rxn

        self.model.objective_direction = direction
        self.solution = self.model.optimize()

        if self.solution.status != "optimal":
            logger.warning(f"Optimization status: {self.solution.status}")

        return self.solution.fluxes

    def flux_variability(
        self,
        reactions: list[str] | None = None,
        fraction_of_optimum: float = 1.0,
        loopless: bool = False,
    ) -> pd.DataFrame:
        """Perform Flux Variability Analysis (FVA).

        Determines the minimum and maximum flux through each reaction
        while maintaining a fraction of the optimal objective value.

        Parameters
        ----------
        reactions : list[str], optional
            Reactions to analyze. If None, analyzes all reactions.
        fraction_of_optimum : float, default=1.0
            Fraction of optimal objective to maintain (0 to 1).
        loopless : bool, default=False
            If True, use loopless FVA to avoid thermodynamically
            infeasible loops.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'minimum' and 'maximum' columns for each reaction.
        """
        if reactions is None:
            reaction_list = self.model.reactions
        else:
            reaction_list = [self.model.reactions.get_by_id(r) for r in reactions]

        fva_result = cobra.flux_analysis.flux_variability_analysis(
            self.model,
            reaction_list=reaction_list,
            fraction_of_optimum=fraction_of_optimum,
            loopless=loopless,
        )

        return fva_result

    def set_bounds(
        self,
        reaction_id: str,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        """Set flux bounds for a reaction.

        Parameters
        ----------
        reaction_id : str
            The reaction to modify.
        lower : float, optional
            Lower bound. If None, keeps current.
        upper : float, optional
            Upper bound. If None, keeps current.
        """
        rxn = self.model.reactions.get_by_id(reaction_id)

        # Store original bounds
        if reaction_id not in self._original_bounds:
            self._original_bounds[reaction_id] = (rxn.lower_bound, rxn.upper_bound)

        if lower is not None:
            rxn.lower_bound = lower
        if upper is not None:
            rxn.upper_bound = upper

    def reset_bounds(self, reaction_id: str | None = None) -> None:
        """Reset reaction bounds to original values.

        Parameters
        ----------
        reaction_id : str, optional
            Reaction to reset. If None, resets all modified reactions.
        """
        if reaction_id is not None:
            if reaction_id in self._original_bounds:
                rxn = self.model.reactions.get_by_id(reaction_id)
                lower, upper = self._original_bounds[reaction_id]
                rxn.lower_bound = lower
                rxn.upper_bound = upper
                del self._original_bounds[reaction_id]
        else:
            for rxn_id, (lower, upper) in self._original_bounds.items():
                rxn = self.model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = lower
                rxn.upper_bound = upper
            self._original_bounds.clear()

    def knockout(self, reaction_ids: str | list[str]) -> pd.Series:
        """Perform reaction knockout analysis.

        Parameters
        ----------
        reaction_ids : str or list[str]
            Reaction(s) to knock out.

        Returns
        -------
        pd.Series
            Flux distribution after knockout.
        """
        if isinstance(reaction_ids, str):
            reaction_ids = [reaction_ids]

        with self.model:
            for rxn_id in reaction_ids:
                rxn = self.model.reactions.get_by_id(rxn_id)
                rxn.knock_out()

            return self.optimize()

    def gene_knockout(self, gene_ids: str | list[str]) -> pd.Series:
        """Perform gene knockout analysis.

        Parameters
        ----------
        gene_ids : str or list[str]
            Gene(s) to knock out.

        Returns
        -------
        pd.Series
            Flux distribution after knockout.
        """
        if isinstance(gene_ids, str):
            gene_ids = [gene_ids]

        with self.model:
            for gene_id in gene_ids:
                gene = self.model.genes.get_by_id(gene_id)
                gene.knock_out()

            return self.optimize()

    def parsimonious_fba(self, fraction_of_optimum: float = 1.0) -> pd.Series:
        """Perform parsimonious FBA (pFBA).

        First optimizes the objective, then minimizes total flux
        while maintaining a fraction of the optimal objective.

        Parameters
        ----------
        fraction_of_optimum : float, default=1.0
            Fraction of optimal objective to maintain.

        Returns
        -------
        pd.Series
            Flux distribution with minimal total flux.
        """
        pfba_solution = cobra.flux_analysis.pfba(
            self.model, fraction_of_optimum=fraction_of_optimum
        )
        self.solution = pfba_solution
        return pfba_solution.fluxes

    def get_exchange_fluxes(self) -> pd.Series:
        """Get fluxes for exchange reactions.

        Returns
        -------
        pd.Series
            Exchange reaction fluxes.
        """
        if self.solution is None:
            self.optimize()

        exchange_ids = [rxn.id for rxn in self.model.exchanges]
        return self.solution.fluxes[exchange_ids]

    def summary(self) -> str:
        """Get a summary of the FBA solution.

        Returns
        -------
        str
            Summary string with objective value and key fluxes.
        """
        if self.solution is None:
            return "No solution computed yet. Run optimize() first."

        lines = [
            f"FBA Solution Summary",
            f"=" * 40,
            f"Status: {self.solution.status}",
            f"Objective value: {self.solution.objective_value:.6f}",
            f"",
            f"Top 10 reactions by flux:",
        ]

        top_fluxes = self.solution.fluxes.abs().nlargest(10)
        for rxn_id, flux in top_fluxes.items():
            actual_flux = self.solution.fluxes[rxn_id]
            lines.append(f"  {rxn_id}: {actual_flux:.4f}")

        return "\n".join(lines)


def compute_yield(
    model: cobra.Model,
    product_reaction: str,
    substrate_reaction: str,
    substrate_uptake: float = 10.0,
) -> float:
    """Compute theoretical yield of product from substrate.

    Parameters
    ----------
    model : cobra.Model
        Metabolic model.
    product_reaction : str
        Exchange reaction for product.
    substrate_reaction : str
        Exchange reaction for substrate (uptake).
    substrate_uptake : float, default=10.0
        Substrate uptake rate.

    Returns
    -------
    float
        Theoretical yield (product/substrate).
    """
    with model:
        # Set substrate uptake
        model.reactions.get_by_id(substrate_reaction).lower_bound = -substrate_uptake

        # Maximize product
        model.objective = product_reaction
        solution = model.optimize()

        if solution.status == "optimal":
            product_flux = solution.fluxes[product_reaction]
            return product_flux / substrate_uptake
        else:
            return 0.0


def find_blocked_reactions(model: cobra.Model) -> list[str]:
    """Find reactions that cannot carry flux.

    Parameters
    ----------
    model : cobra.Model
        Metabolic model.

    Returns
    -------
    list[str]
        List of blocked reaction IDs.
    """
    blocked = cobra.flux_analysis.find_blocked_reactions(model)
    return list(blocked)


def find_essential_reactions(
    model: cobra.Model,
    threshold: float = 0.01,
) -> list[str]:
    """Find reactions essential for growth.

    A reaction is essential if its knockout reduces the objective
    below the threshold fraction of wild-type.

    Parameters
    ----------
    model : cobra.Model
        Metabolic model.
    threshold : float, default=0.01
        Minimum fraction of wild-type objective for viability.

    Returns
    -------
    list[str]
        List of essential reaction IDs.
    """
    # Get wild-type objective
    wt_solution = model.optimize()
    wt_objective = wt_solution.objective_value

    essential = []

    for rxn in model.reactions:
        if rxn.boundary:
            continue

        with model:
            rxn.knock_out()
            try:
                ko_solution = model.optimize()
                if ko_solution.status != "optimal":
                    essential.append(rxn.id)
                elif ko_solution.objective_value < threshold * wt_objective:
                    essential.append(rxn.id)
            except Exception:
                essential.append(rxn.id)

    return essential


def apply_media(
    model: cobra.Model,
    media: dict[str, float],
    default_uptake: float = 0.0,
) -> cobra.Model:
    """Apply media constraints to model.

    Parameters
    ----------
    model : cobra.Model
        Metabolic model.
    media : dict[str, float]
        Dictionary mapping exchange reaction IDs to uptake rates
        (positive values = uptake allowed).
    default_uptake : float, default=0.0
        Default uptake rate for exchanges not in media dict.

    Returns
    -------
    cobra.Model
        Model with updated exchange bounds.
    """
    model = model.copy()

    for rxn in model.exchanges:
        if rxn.id in media:
            rxn.lower_bound = -media[rxn.id]
        else:
            rxn.lower_bound = -default_uptake

    return model
