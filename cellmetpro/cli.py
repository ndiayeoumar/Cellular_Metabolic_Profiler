"""Command-line interface for CellMetPro.

This module provides CLI commands for running metabolic analysis
pipelines from the command line.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cellmetpro")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="cellmetpro",
        description="CellMetPro - Cellular Metabolic Profiler for scRNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run COMPASS analysis on expression data
  cellmetpro run expression.h5ad -m human -o results/

  # Run with custom model file
  cellmetpro run data.csv -m /path/to/model.xml -o output/

  # Run with microclustering for large datasets
  cellmetpro run large_data.h5ad --microcluster --cells-per-cluster 100

  # Launch interactive dashboard
  cellmetpro dashboard results/
""",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run COMPASS metabolic analysis pipeline",
        description="Run COMPASS algorithm to score metabolic reactions from scRNA-seq data",
    )
    run_parser.add_argument(
        "input",
        type=Path,
        help="Input file (h5ad, csv, or tsv format)",
    )
    run_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)",
    )
    run_parser.add_argument(
        "-m", "--model",
        type=str,
        default="human",
        help="Metabolic model: 'human', 'mouse', 'recon2', or path to SBML/JSON file",
    )
    run_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize expression data before analysis",
    )
    run_parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Target sum for normalization (default: 10000)",
    )

    # COMPASS parameters
    run_parser.add_argument(
        "--beta",
        type=float,
        default=0.95,
        help="COMPASS beta parameter (default: 0.95)",
    )
    run_parser.add_argument(
        "--lambda",
        dest="lambda_penalty",
        type=float,
        default=0.0,
        help="Penalty smoothing lambda (0-1, default: 0)",
    )
    run_parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="Number of neighbors for penalty smoothing (default: 30)",
    )

    # Performance options
    run_parser.add_argument(
        "-j", "--n-processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    run_parser.add_argument(
        "--microcluster",
        action="store_true",
        help="Use microclustering for large datasets",
    )
    run_parser.add_argument(
        "--cells-per-cluster",
        type=int,
        default=100,
        help="Target cells per microcluster (default: 100)",
    )
    run_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Cache optimization results for faster reruns",
    )

    # Output options
    run_parser.add_argument(
        "--compute-exchange",
        action="store_true",
        help="Also compute uptake/secretion scores",
    )
    run_parser.add_argument(
        "--output-format",
        choices=["csv", "parquet", "h5ad"],
        default="csv",
        help="Output file format (default: csv)",
    )

    # Dashboard command
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Launch interactive Streamlit dashboard",
    )
    dash_parser.add_argument(
        "results",
        type=Path,
        help="Path to results directory from 'run' command",
    )
    dash_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8501,
        help="Port for Streamlit server (default: 8501)",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a metabolic model",
    )
    info_parser.add_argument(
        "model",
        type=str,
        help="Model name or path to model file",
    )

    return parser


def run_analysis(args: argparse.Namespace) -> int:
    """Run the COMPASS metabolic analysis pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    import pandas as pd

    from cellmetpro.core import (
        CompassConfig,
        CompassScorer,
        DataLoader,
        microcluster,
        normalize_expression,
        to_dataframe,
        unpool_results,
    )
    from cellmetpro.models import load_gem

    logger.info(f"CellMetPro COMPASS Analysis")
    logger.info(f"=" * 50)
    logger.info(f"Input: {args.input}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load expression data
    logger.info("Loading expression data...")
    loader = DataLoader(args.input)
    adata = loader.load()
    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

    # Normalize if requested
    if args.normalize:
        logger.info("Normalizing expression data...")
        adata = normalize_expression(adata, target_sum=args.target_sum)

    # Convert to DataFrame (genes x cells)
    expression_df = to_dataframe(adata, genes_as_rows=True)

    # Load metabolic model
    logger.info(f"Loading metabolic model: {args.model}")
    model = load_gem(args.model)
    logger.info(
        f"Model: {model.id} ({len(model.reactions)} reactions, "
        f"{len(model.metabolites)} metabolites, {len(model.genes)} genes)"
    )

    # Configure COMPASS
    config = CompassConfig(
        beta=args.beta,
        lambda_penalty=args.lambda_penalty,
        n_neighbors=args.n_neighbors,
        n_processes=args.n_processes,
    )

    # Handle microclustering for large datasets
    microcluster_result = None
    if args.microcluster:
        from cellmetpro.core import MicroclusterConfig

        logger.info(f"Microclustering cells (target: {args.cells_per_cluster} cells/cluster)...")
        mc_config = MicroclusterConfig(cells_per_cluster=args.cells_per_cluster)
        microcluster_result = microcluster(expression_df, mc_config)
        logger.info(f"Created {microcluster_result.n_clusters} microclusters")

        # Use pooled expression
        expression_df = microcluster_result.pooled_expression

    # Run COMPASS
    logger.info("Running COMPASS algorithm...")
    scorer = CompassScorer(model, expression_df, config)
    result = scorer.score(compute_exchange=args.compute_exchange)

    # Unpool results if microclustering was used
    if microcluster_result is not None:
        logger.info("Unpooling results to individual cells...")
        result.reaction_penalties = unpool_results(
            result.reaction_penalties, microcluster_result
        )
        result.reaction_scores = unpool_results(
            result.reaction_scores, microcluster_result
        )

    # Save results
    logger.info(f"Saving results to {args.output}/")

    if args.output_format == "csv":
        result.reaction_penalties.to_csv(args.output / "reaction_penalties.csv")
        result.reaction_scores.to_csv(args.output / "reaction_scores.csv")
        if result.uptake_scores is not None:
            result.uptake_scores.to_csv(args.output / "uptake_scores.csv")
        if result.secretion_scores is not None:
            result.secretion_scores.to_csv(args.output / "secretion_scores.csv")
    elif args.output_format == "parquet":
        result.reaction_penalties.to_parquet(args.output / "reaction_penalties.parquet")
        result.reaction_scores.to_parquet(args.output / "reaction_scores.parquet")
        if result.uptake_scores is not None:
            result.uptake_scores.to_parquet(args.output / "uptake_scores.parquet")
        if result.secretion_scores is not None:
            result.secretion_scores.to_parquet(args.output / "secretion_scores.parquet")
    elif args.output_format == "h5ad":
        import anndata as ad

        # Store results in AnnData format
        result_adata = ad.AnnData(result.reaction_scores.T)
        result_adata.layers["penalties"] = result.reaction_penalties.T.values
        result_adata.write(args.output / "compass_results.h5ad")

    # Save config
    import json

    config_dict = {
        "input": str(args.input),
        "model": args.model,
        "beta": config.beta,
        "lambda_penalty": config.lambda_penalty,
        "n_neighbors": config.n_neighbors,
        "microcluster": args.microcluster,
        "cells_per_cluster": args.cells_per_cluster if args.microcluster else None,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "n_reactions": len(result.reaction_scores),
    }
    with open(args.output / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Analysis complete!")
    logger.info(f"Results saved to: {args.output}/")
    logger.info(f"  - reaction_penalties.{args.output_format}")
    logger.info(f"  - reaction_scores.{args.output_format}")
    if args.compute_exchange:
        logger.info(f"  - uptake_scores.{args.output_format}")
        logger.info(f"  - secretion_scores.{args.output_format}")

    return 0


def run_dashboard(args: argparse.Namespace) -> int:
    """Launch the interactive dashboard.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    import subprocess

    if not args.results.exists():
        logger.error(f"Results directory not found: {args.results}")
        return 1

    logger.info(f"Launching dashboard for results in {args.results}")
    logger.info(f"Running on port {args.port}")

    # Check if streamlit is available
    try:
        import streamlit
    except ImportError:
        logger.error(
            "Streamlit not installed. Install with: pip install cellmetpro[dashboard]"
        )
        return 1

    # Get path to dashboard script
    from cellmetpro.visualization import dashboard

    dashboard_path = Path(dashboard.__file__)

    # Run streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(args.port),
        "--",
        str(args.results),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Dashboard failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")

    return 0


def show_model_info(args: argparse.Namespace) -> int:
    """Show information about a metabolic model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    from cellmetpro.models import load_gem, get_subsystem_reactions

    try:
        model = load_gem(args.model)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    print(f"\nModel Information: {model.id}")
    print("=" * 50)
    print(f"Name: {model.name or 'N/A'}")
    print(f"Reactions: {len(model.reactions)}")
    print(f"Metabolites: {len(model.metabolites)}")
    print(f"Genes: {len(model.genes)}")
    print(f"Exchange reactions: {len(model.exchanges)}")

    # Count reactions with GPR rules
    n_with_gpr = sum(1 for r in model.reactions if r.gene_reaction_rule)
    print(f"Reactions with GPR rules: {n_with_gpr}")

    # Subsystems
    subsystems = get_subsystem_reactions(model)
    print(f"Subsystems: {len(subsystems)}")

    if subsystems:
        print("\nTop 10 subsystems by reaction count:")
        sorted_subsystems = sorted(
            subsystems.items(), key=lambda x: len(x[1]), reverse=True
        )[:10]
        for name, rxns in sorted_subsystems:
            print(f"  {name}: {len(rxns)} reactions")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.version:
        from cellmetpro import __version__

        print(f"cellmetpro {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "run":
            return run_analysis(args)
        elif args.command == "dashboard":
            return run_dashboard(args)
        elif args.command == "info":
            return show_model_info(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
