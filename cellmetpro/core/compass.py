"""COMPASS algorithm implementation for metabolic scoring.

COMPASS (Characterizing Cell states through metabolic Profiling of the
Transcriptome) integrates scRNA-seq data with genome-scale metabolic
models to infer metabolic activity at single-cell resolution.

This implementation uses COBRApy for constraint-based modeling, providing
an open-source alternative to the original Gurobi-based implementation.

References:
    Wagner et al. (2021) "Metabolic modeling of single Th17 cells reveals
    regulators of autoimmunity" Cell.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import pandas as pd
from scipy.sparse import issparse

if TYPE_CHECKING:
    import anndata as ad
    import cobra

logger = logging.getLogger(__name__)

# COMPASS algorithm constants
BETA = 0.95  # Fraction of maximum flux to maintain during optimization
EXCHANGE_LIMIT = 1000.0  # Maximum flux for exchange reactions
PENALTY_EPSILON = 1e-6  # Small value to avoid division by zero


@dataclass
class CompassConfig:
    """Configuration for COMPASS algorithm.

    Attributes
    ----------
    beta : float
        Fraction of maximum flux to maintain (default: 0.95).
    exchange_limit : float
        Maximum flux for exchange reactions (default: 1000.0).
    and_function : str
        Function for AND operations in GPR: 'min', 'mean', 'median'.
    or_function : str
        Function for OR operations in GPR: 'max', 'sum', 'mean'.
    lambda_penalty : float
        Weight for penalty smoothing (0 to 1). Higher values give more weight
        to neighborhood-based penalties.
    n_neighbors : int
        Number of neighbors for KNN-based penalty smoothing.
    n_processes : int
        Number of parallel processes for computation.
    solver : str
        LP solver to use ('glpk', 'cplex', 'gurobi').
    """

    beta: float = BETA
    exchange_limit: float = EXCHANGE_LIMIT
    and_function: Literal["min", "mean", "median"] = "min"
    or_function: Literal["max", "sum", "mean"] = "sum"
    lambda_penalty: float = 0.0
    n_neighbors: int = 30
    n_processes: int = 1
    solver: str = "glpk"


@dataclass
class CompassResult:
    """Results from COMPASS analysis.

    Attributes
    ----------
    reaction_penalties : pd.DataFrame
        Reaction penalty scores (reactions x cells). Higher = less likely active.
    reaction_scores : pd.DataFrame
        Reaction consistency scores after optimization (reactions x cells).
    uptake_scores : pd.DataFrame | None
        Metabolite uptake scores (metabolites x cells).
    secretion_scores : pd.DataFrame | None
        Metabolite secretion scores (metabolites x cells).
    config : CompassConfig
        Configuration used for the analysis.
    """

    reaction_penalties: pd.DataFrame
    reaction_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    uptake_scores: pd.DataFrame | None = None
    secretion_scores: pd.DataFrame | None = None
    config: CompassConfig = field(default_factory=CompassConfig)


class CompassScorer:
    """Compute metabolic reaction scores using the COMPASS algorithm.

    COMPASS evaluates the consistency of each metabolic reaction with
    a cell's gene expression profile. It uses constraint-based modeling
    to determine which reactions are feasible given expression data.

    Parameters
    ----------
    model : cobra.Model
        A genome-scale metabolic model (GEM).
    gene_expression : pd.DataFrame | ad.AnnData
        Gene expression matrix. If DataFrame: genes x cells.
        If AnnData: cells x genes (will be transposed internally).
    config : CompassConfig, optional
        Configuration parameters for the algorithm.

    Attributes
    ----------
    reaction_penalties : pd.DataFrame
        Computed reaction penalties (reactions x cells).
    reaction_scores : pd.DataFrame
        Final reaction consistency scores (reactions x cells).

    Examples
    --------
    >>> import cobra
    >>> from cellmetpro.core.compass import CompassScorer, CompassConfig
    >>> model = cobra.io.read_sbml_model("model.xml")
    >>> expression = pd.read_csv("expression.csv", index_col=0)
    >>> scorer = CompassScorer(model, expression)
    >>> result = scorer.score()
    >>> print(result.reaction_scores.head())
    """

    def __init__(
        self,
        model: cobra.Model,
        gene_expression: pd.DataFrame | ad.AnnData,
        config: CompassConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or CompassConfig()
        self._reaction_penalties: pd.DataFrame | None = None
        self._reaction_scores: pd.DataFrame | None = None

        # Process gene expression input
        self.gene_expression = self._process_expression_input(gene_expression)
        self.cell_names = list(self.gene_expression.columns)
        self.gene_names = list(self.gene_expression.index)

        # Normalize gene names to uppercase for matching
        self._expression_upper = self.gene_expression.copy()
        self._expression_upper.index = self._expression_upper.index.str.upper()

        # Build gene-reaction mapping
        self._build_gpr_mapping()

    def _process_expression_input(
        self, expr: pd.DataFrame | ad.AnnData
    ) -> pd.DataFrame:
        """Convert expression input to genes x cells DataFrame."""
        # Check if AnnData
        try:
            import anndata as ad

            if isinstance(expr, ad.AnnData):
                # AnnData is cells x genes, we need genes x cells
                if issparse(expr.X):
                    data = expr.X.toarray().T
                else:
                    data = expr.X.T
                return pd.DataFrame(
                    data, index=expr.var_names, columns=expr.obs_names
                )
        except ImportError:
            pass

        if isinstance(expr, pd.DataFrame):
            return expr

        raise TypeError(
            f"gene_expression must be pd.DataFrame or AnnData, got {type(expr)}"
        )

    def _build_gpr_mapping(self) -> None:
        """Build mapping from reactions to gene expression evaluation functions."""
        self._reaction_genes: dict[str, list[str]] = {}
        self._reaction_gpr: dict[str, str] = {}

        for rxn in self.model.reactions:
            if rxn.genes:
                self._reaction_genes[rxn.id] = [g.id for g in rxn.genes]
                self._reaction_gpr[rxn.id] = rxn.gene_reaction_rule

    @property
    def reaction_penalties(self) -> pd.DataFrame:
        """Get computed reaction penalties."""
        if self._reaction_penalties is None:
            self._reaction_penalties = self.compute_reaction_penalties()
        return self._reaction_penalties

    @property
    def reaction_scores(self) -> pd.DataFrame:
        """Get computed reaction scores."""
        if self._reaction_scores is None:
            result = self.score()
            self._reaction_scores = result.reaction_scores
        return self._reaction_scores

    def compute_reaction_penalties(self) -> pd.DataFrame:
        """Compute reaction penalties from gene expression.

        Penalties are computed by evaluating Gene-Protein-Reaction (GPR)
        rules using the expression values. Higher penalties indicate
        reactions that are less likely to be active.

        Returns
        -------
        pd.DataFrame
            Reaction penalties matrix (reactions x cells).
        """
        logger.info("Computing reaction penalties from gene expression...")

        # Get reactions with GPR rules
        reactions_with_gpr = [
            rxn for rxn in self.model.reactions if rxn.gene_reaction_rule
        ]

        if not reactions_with_gpr:
            logger.warning("No reactions with GPR rules found in model")
            return pd.DataFrame()

        # Compute expression for each reaction
        reaction_expression = {}

        for rxn in reactions_with_gpr:
            expr_values = self._evaluate_gpr(
                rxn.gene_reaction_rule, self._expression_upper
            )
            reaction_expression[rxn.id] = expr_values

        # Create DataFrame
        reaction_expr_df = pd.DataFrame(reaction_expression).T
        reaction_expr_df.columns = self.cell_names

        # Convert expression to penalties (higher expression = lower penalty)
        # Use log transform and inversion
        penalties = self._expression_to_penalty(reaction_expr_df)

        logger.info(
            f"Computed penalties for {len(penalties)} reactions across "
            f"{len(self.cell_names)} cells"
        )

        # Apply penalty smoothing if lambda > 0
        if self.config.lambda_penalty > 0:
            penalties = self._smooth_penalties(penalties)

        self._reaction_penalties = penalties
        return penalties

    def _evaluate_gpr(self, gpr_rule: str, expression: pd.DataFrame) -> np.ndarray:
        """Evaluate a GPR rule using expression data.

        Parameters
        ----------
        gpr_rule : str
            Gene-Protein-Reaction rule (e.g., "GENE1 and (GENE2 or GENE3)").
        expression : pd.DataFrame
            Expression matrix with uppercase gene names as index.

        Returns
        -------
        np.ndarray
            Expression values for each cell.
        """
        if not gpr_rule or gpr_rule.strip() == "":
            return np.zeros(expression.shape[1])

        # Define aggregation functions
        and_funcs = {"min": np.min, "mean": np.mean, "median": np.median}
        or_funcs = {"max": np.max, "sum": np.sum, "mean": np.mean}

        and_func = and_funcs[self.config.and_function]
        or_func = or_funcs[self.config.or_function]

        # Parse and evaluate the GPR rule
        return self._evaluate_gpr_recursive(
            gpr_rule.upper(), expression, and_func, or_func
        )

    def _evaluate_gpr_recursive(
        self,
        gpr_rule: str,
        expression: pd.DataFrame,
        and_func: Callable,
        or_func: Callable,
    ) -> np.ndarray:
        """Recursively evaluate GPR rule."""
        gpr_rule = gpr_rule.strip()

        # Remove outer parentheses
        while gpr_rule.startswith("(") and gpr_rule.endswith(")"):
            # Check if these are matching parentheses
            depth = 0
            matching = True
            for i, char in enumerate(gpr_rule[:-1]):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                if depth == 0 and i < len(gpr_rule) - 2:
                    matching = False
                    break
            if matching:
                gpr_rule = gpr_rule[1:-1].strip()
            else:
                break

        # Find top-level OR operators
        or_parts = self._split_at_operator(gpr_rule, " OR ")
        if len(or_parts) > 1:
            values = [
                self._evaluate_gpr_recursive(part, expression, and_func, or_func)
                for part in or_parts
            ]
            return or_func(np.array(values), axis=0)

        # Find top-level AND operators
        and_parts = self._split_at_operator(gpr_rule, " AND ")
        if len(and_parts) > 1:
            values = [
                self._evaluate_gpr_recursive(part, expression, and_func, or_func)
                for part in and_parts
            ]
            return and_func(np.array(values), axis=0)

        # Base case: single gene
        gene = gpr_rule.strip()
        if gene in expression.index:
            return expression.loc[gene].values.astype(float)
        else:
            # Gene not found - return zeros
            return np.zeros(expression.shape[1])

    def _split_at_operator(self, rule: str, operator: str) -> list[str]:
        """Split GPR rule at operator, respecting parentheses."""
        parts = []
        current = ""
        depth = 0

        i = 0
        while i < len(rule):
            if rule[i] == "(":
                depth += 1
                current += rule[i]
            elif rule[i] == ")":
                depth -= 1
                current += rule[i]
            elif depth == 0 and rule[i:].upper().startswith(operator):
                if current.strip():
                    parts.append(current.strip())
                current = ""
                i += len(operator) - 1
            else:
                current += rule[i]
            i += 1

        if current.strip():
            parts.append(current.strip())

        return parts if len(parts) > 1 else [rule]

    def _expression_to_penalty(self, expression: pd.DataFrame) -> pd.DataFrame:
        """Convert expression values to penalty scores.

        Higher expression leads to lower penalty (reaction more likely).

        Parameters
        ----------
        expression : pd.DataFrame
            Reaction expression values.

        Returns
        -------
        pd.DataFrame
            Penalty scores.
        """
        # Add small epsilon to avoid log(0)
        expr_safe = expression + PENALTY_EPSILON

        # Log transform
        log_expr = np.log1p(expr_safe)

        # Invert: high expression = low penalty
        # Normalize to [0, 1] range per reaction, then invert
        max_vals = log_expr.max(axis=1)
        max_vals[max_vals == 0] = 1  # Avoid division by zero

        normalized = log_expr.div(max_vals, axis=0)
        penalties = 1 - normalized

        return penalties

    def _smooth_penalties(self, penalties: pd.DataFrame) -> pd.DataFrame:
        """Apply KNN-based penalty smoothing.

        Blends each cell's penalties with its neighbors' penalties.

        Parameters
        ----------
        penalties : pd.DataFrame
            Raw penalty scores.

        Returns
        -------
        pd.DataFrame
            Smoothed penalty scores.
        """
        from sklearn.neighbors import NearestNeighbors

        logger.info(
            f"Smoothing penalties with {self.config.n_neighbors} neighbors, "
            f"lambda={self.config.lambda_penalty}"
        )

        # Use expression for neighbor computation
        expr_matrix = self.gene_expression.T.values  # cells x genes

        # Fit KNN
        n_neighbors = min(self.config.n_neighbors + 1, len(self.cell_names))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(expr_matrix)

        # Get neighbors
        distances, indices = knn.kneighbors(expr_matrix)

        # Compute smoothed penalties
        penalties_array = penalties.values.T  # cells x reactions
        smoothed = np.zeros_like(penalties_array)

        for i in range(len(self.cell_names)):
            neighbor_indices = indices[i, 1:]  # Exclude self
            neighbor_penalties = penalties_array[neighbor_indices].mean(axis=0)

            # Blend with lambda
            smoothed[i] = (
                1 - self.config.lambda_penalty
            ) * penalties_array[i] + self.config.lambda_penalty * neighbor_penalties

        smoothed_df = pd.DataFrame(
            smoothed.T, index=penalties.index, columns=penalties.columns
        )

        return smoothed_df

    def optimize_reactions(
        self, penalties: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Run COMPASS optimization for each reaction.

        For each reaction and each cell:
        1. Maximize the reaction's flux
        2. Constrain the reaction to BETA * max_flux
        3. Minimize total weighted penalty across all reactions
        4. Record the minimum penalty as the reaction's score

        Parameters
        ----------
        penalties : pd.DataFrame, optional
            Reaction penalties. If None, computes from expression.

        Returns
        -------
        pd.DataFrame
            Reaction consistency scores (reactions x cells).
            Higher scores indicate reactions that are LESS consistent
            with the cell's expression (following original COMPASS convention).
        """
        if penalties is None:
            penalties = self.reaction_penalties

        logger.info("Running COMPASS optimization...")

        import cobra

        # Prepare model
        model = self.model.copy()

        # Get non-exchange reactions for optimization
        internal_reactions = [
            rxn.id
            for rxn in model.reactions
            if rxn.id in penalties.index and not rxn.boundary
        ]

        logger.info(f"Optimizing {len(internal_reactions)} reactions")

        # Run optimization for each cell
        scores = {}

        if self.config.n_processes > 1:
            # Parallel processing
            scores = self._optimize_parallel(model, penalties, internal_reactions)
        else:
            # Sequential processing
            for cell_idx, cell_name in enumerate(self.cell_names):
                if cell_idx % 10 == 0:
                    logger.info(f"Processing cell {cell_idx + 1}/{len(self.cell_names)}")

                cell_penalties = penalties[cell_name]
                cell_scores = self._optimize_single_cell(
                    model, cell_penalties, internal_reactions
                )
                scores[cell_name] = cell_scores

        scores_df = pd.DataFrame(scores)
        logger.info("COMPASS optimization complete")

        return scores_df

    def _optimize_single_cell(
        self,
        model: cobra.Model,
        penalties: pd.Series,
        reaction_ids: list[str],
    ) -> dict[str, float]:
        """Optimize reactions for a single cell.

        Parameters
        ----------
        model : cobra.Model
            Metabolic model.
        penalties : pd.Series
            Penalty values for this cell.
        reaction_ids : list[str]
            Reactions to optimize.

        Returns
        -------
        dict[str, float]
            Reaction ID to score mapping.
        """
        import cobra

        scores = {}
        model = model.copy()

        # Cache for max fluxes
        max_flux_cache = {}

        for rxn_id in reaction_ids:
            try:
                rxn = model.reactions.get_by_id(rxn_id)

                # Step 1: Maximize this reaction's flux
                if rxn_id not in max_flux_cache:
                    model.objective = rxn
                    model.objective_direction = "max"

                    with model:
                        solution = model.optimize()
                        if solution.status == "optimal":
                            max_flux = abs(solution.objective_value)
                        else:
                            max_flux = 0
                    max_flux_cache[rxn_id] = max_flux
                else:
                    max_flux = max_flux_cache[rxn_id]

                if max_flux < 1e-9:
                    # Reaction cannot carry flux
                    scores[rxn_id] = penalties.get(rxn_id, 1.0)
                    continue

                # Step 2: Constrain reaction to BETA * max_flux and minimize penalty
                with model:
                    # Set constraint
                    constraint_value = self.config.beta * max_flux
                    if rxn.upper_bound > 0:
                        rxn.lower_bound = max(rxn.lower_bound, constraint_value)
                    else:
                        rxn.upper_bound = min(rxn.upper_bound, -constraint_value)

                    # Create penalty objective
                    # Minimize sum of penalty * |flux| for all reactions
                    objective_dict = {}
                    for r in model.reactions:
                        if r.id in penalties.index:
                            penalty = penalties[r.id]
                            objective_dict[r.forward_variable] = penalty
                            objective_dict[r.reverse_variable] = penalty

                    model.objective = objective_dict
                    model.objective_direction = "min"

                    solution = model.optimize()

                    if solution.status == "optimal":
                        scores[rxn_id] = solution.objective_value
                    else:
                        scores[rxn_id] = penalties.get(rxn_id, 1.0)

            except Exception as e:
                logger.debug(f"Error optimizing {rxn_id}: {e}")
                scores[rxn_id] = penalties.get(rxn_id, 1.0)

        return scores

    def _optimize_parallel(
        self,
        model: cobra.Model,
        penalties: pd.DataFrame,
        reaction_ids: list[str],
    ) -> dict[str, dict[str, float]]:
        """Run optimization in parallel across cells."""
        import pickle

        # Serialize model for passing to workers
        model_str = cobra.io.to_json(model)

        scores = {}

        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            futures = {}
            for cell_name in self.cell_names:
                cell_penalties = penalties[cell_name]
                future = executor.submit(
                    _optimize_cell_worker,
                    model_str,
                    cell_penalties.to_dict(),
                    reaction_ids,
                    self.config.beta,
                )
                futures[future] = cell_name

            for future in as_completed(futures):
                cell_name = futures[future]
                try:
                    scores[cell_name] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {cell_name}: {e}")
                    scores[cell_name] = {rxn: 1.0 for rxn in reaction_ids}

        return scores

    def score(self, compute_exchange: bool = False) -> CompassResult:
        """Compute full COMPASS scores.

        This is the main entry point for running the complete COMPASS
        algorithm, including penalty computation and optimization.

        Parameters
        ----------
        compute_exchange : bool, default=False
            If True, also compute uptake and secretion scores for
            exchange reactions.

        Returns
        -------
        CompassResult
            Complete results including penalties, scores, and optionally
            exchange scores.
        """
        logger.info("Starting COMPASS analysis...")

        # Compute penalties
        penalties = self.compute_reaction_penalties()

        # Run optimization
        scores = self.optimize_reactions(penalties)

        # Compute exchange scores if requested
        uptake = None
        secretion = None
        if compute_exchange:
            uptake, secretion = self._compute_exchange_scores(penalties)

        result = CompassResult(
            reaction_penalties=penalties,
            reaction_scores=scores,
            uptake_scores=uptake,
            secretion_scores=secretion,
            config=self.config,
        )

        logger.info("COMPASS analysis complete")
        return result

    def _compute_exchange_scores(
        self, penalties: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute uptake and secretion scores for exchange reactions.

        Parameters
        ----------
        penalties : pd.DataFrame
            Reaction penalties.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Uptake scores and secretion scores.
        """
        logger.info("Computing exchange reaction scores...")

        import cobra

        model = self.model.copy()

        # Get exchange reactions
        exchange_rxns = [rxn for rxn in model.reactions if rxn.boundary]

        uptake_scores = {}
        secretion_scores = {}

        for cell_name in self.cell_names:
            cell_penalties = penalties[cell_name] if cell_name in penalties.columns else pd.Series()

            uptake_cell = {}
            secretion_cell = {}

            for rxn in exchange_rxns:
                metabolite_id = list(rxn.metabolites.keys())[0].id if rxn.metabolites else rxn.id

                with model:
                    # Secretion: maximize forward flux
                    model.objective = rxn
                    model.objective_direction = "max"
                    sol = model.optimize()

                    if sol.status == "optimal" and sol.objective_value > 1e-9:
                        secretion_cell[metabolite_id] = sol.objective_value
                    else:
                        secretion_cell[metabolite_id] = 0.0

                with model:
                    # Uptake: minimize (maximize negative)
                    model.objective = rxn
                    model.objective_direction = "min"
                    sol = model.optimize()

                    if sol.status == "optimal" and sol.objective_value < -1e-9:
                        uptake_cell[metabolite_id] = abs(sol.objective_value)
                    else:
                        uptake_cell[metabolite_id] = 0.0

            uptake_scores[cell_name] = uptake_cell
            secretion_scores[cell_name] = secretion_cell

        uptake_df = pd.DataFrame(uptake_scores)
        secretion_df = pd.DataFrame(secretion_scores)

        return uptake_df, secretion_df


def _optimize_cell_worker(
    model_json: str,
    penalties_dict: dict[str, float],
    reaction_ids: list[str],
    beta: float,
) -> dict[str, float]:
    """Worker function for parallel cell optimization.

    This function is called in a separate process.
    """
    import cobra

    model = cobra.io.from_json(model_json)
    penalties = pd.Series(penalties_dict)

    scores = {}

    for rxn_id in reaction_ids:
        try:
            rxn = model.reactions.get_by_id(rxn_id)

            # Maximize reaction flux
            model.objective = rxn
            model.objective_direction = "max"

            with model:
                solution = model.optimize()
                if solution.status == "optimal":
                    max_flux = abs(solution.objective_value)
                else:
                    max_flux = 0

            if max_flux < 1e-9:
                scores[rxn_id] = penalties.get(rxn_id, 1.0)
                continue

            # Constrain and minimize penalty
            with model:
                constraint_value = beta * max_flux
                if rxn.upper_bound > 0:
                    rxn.lower_bound = max(rxn.lower_bound, constraint_value)
                else:
                    rxn.upper_bound = min(rxn.upper_bound, -constraint_value)

                objective_dict = {}
                for r in model.reactions:
                    if r.id in penalties.index:
                        penalty = penalties[r.id]
                        objective_dict[r.forward_variable] = penalty
                        objective_dict[r.reverse_variable] = penalty

                model.objective = objective_dict
                model.objective_direction = "min"

                solution = model.optimize()

                if solution.status == "optimal":
                    scores[rxn_id] = solution.objective_value
                else:
                    scores[rxn_id] = penalties.get(rxn_id, 1.0)

        except Exception:
            scores[rxn_id] = penalties.get(rxn_id, 1.0)

    return scores


def run_compass(
    model: cobra.Model,
    expression: pd.DataFrame | ad.AnnData,
    config: CompassConfig | None = None,
    compute_exchange: bool = False,
) -> CompassResult:
    """Run COMPASS analysis on gene expression data.

    This is a convenience function that creates a CompassScorer
    and runs the full analysis.

    Parameters
    ----------
    model : cobra.Model
        Genome-scale metabolic model.
    expression : pd.DataFrame or AnnData
        Gene expression data. DataFrame should be genes x cells.
        AnnData should be cells x genes.
    config : CompassConfig, optional
        Algorithm configuration.
    compute_exchange : bool, default=False
        Whether to compute exchange reaction scores.

    Returns
    -------
    CompassResult
        Complete COMPASS results.

    Examples
    --------
    >>> from cellmetpro.core.compass import run_compass, CompassConfig
    >>> from cellmetpro.models import load_gem
    >>> model = load_gem("human")
    >>> result = run_compass(model, expression_data)
    >>> print(result.reaction_scores.head())
    """
    scorer = CompassScorer(model, expression, config)
    return scorer.score(compute_exchange=compute_exchange)
