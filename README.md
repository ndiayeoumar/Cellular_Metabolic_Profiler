# OmicsPilot: Cellular Metabolic Profiler (cellMetPro)

**Visualize and explore cellular metabolic states from scRNA-seq data**

A Python-based tool for biotech researchers to analyze and visualize metabolic profiles at single-cell resolution using Flux Balance Analysis (FBA) and Genome-Scale Metabolic Models (GEMs).

---

## Project Vision

Help researchers **visually explore the metabolic landscape** of their cells to identify biological targets for further investigation. Whether studying immune cells, cancer cells, or stem cells, OmicsPilot provides intuitive visualizations to spot metabolic signatures that distinguish cell populations.

---

## What This Tool Can Do

### Core Analysis Features

| Feature | Description |
|---------|-------------|
| **Metabolic Scoring** | Compute reaction activity scores from gene expression using COMPASS algorithm |
| **Pathway Analysis** | Aggregate reactions into metabolic pathways and metareactions |
| **Cluster Comparison** | Statistical comparison of metabolic states between cell clusters |
| **Dimensionality Reduction** | PCA and UMAP projections of metabolic profiles |

### Visualization Capabilities

- **Metabolic UMAP/t-SNE**: Project cells based on metabolic activity rather than gene expression
- **Pathway Heatmaps**: Compare pathway activity across clusters or conditions
- **Reaction Dot Plots**: Visualize reaction scores with statistical significance
- **Volcano Plots**: Identify differentially active reactions between conditions
- **Metabolite Flow Diagrams**: Track metabolite uptake and secretion patterns
- **Interactive Dashboards**: Explore data with filtering and drill-down capabilities

---

## Biological Applications

### Supported Cell Types
Any cell type with scRNA-seq data with available GEM for the relative organism

### Research Questions This Tool Helps Answer
- Which metabolic pathways are upregulated in disease vs. healthy cells?
- How do metabolic states differ between cell clusters?
- What metabolic signatures define specific cell populations?
- Which reactions could be therapeutic targets?

---

## Data Inputs

| Input Type | Format | Description |
|------------|--------|-------------|
| Gene Expression | `.csv`, `.tsv`, `.h5ad` | TPM-normalized counts matrix |
| Cell Metadata | `.csv`, `.tsv` | Cell annotations, cluster labels, conditions

### Supported Organisms
- Human (Homo sapiens)
- Mouse (Mus musculus)
- Extensible to other organisms with GEM models

---

## Planned Architecture

```
cellmetpro/
├── core/
│   ├── compass.py          # COMPASS algorithm implementation
│   ├── fba.py              # Flux Balance Analysis utilities
│   └── preprocessing.py    # Data loading and normalization
├── analysis/
│   ├── clustering.py       # Metabolic-based clustering
│   ├── differential.py     # Statistical comparisons
│   └── pathway.py          # Pathway aggregation
├── visualization/
│   ├── umap.py             # Dimensionality reduction plots
│   ├── heatmap.py          # Pathway/reaction heatmaps
│   ├── dotplot.py          # Dot plots with statistics
│   ├── volcano.py          # Differential analysis plots
│   └── dashboard.py        # Interactive Streamlit/Dash app
├── models/
│   └── gems/               # Genome-scale metabolic models
└── cli.py                  # Command-line interface
```

---

## 🚀 Roadmap

### Phase 1: Core Infrastructure
- [ ] Data loaders for common scRNA-seq formats (AnnData, Seurat objects)
- [ ] COMPASS algorithm Python implementation
- [ ] Reaction consistency scoring
- [ ] Basic CLI for batch processing

### Phase 2: Visualization Suite
- [ ] Static publication-ready plots (matplotlib/seaborn)
- [ ] Interactive plots (Plotly)
- [ ] Streamlit dashboard for exploration
- [ ] Export to common formats (PNG, SVG, PDF)

### Phase 3: Advanced Analysis
- [ ] Metabolic trajectory analysis
- [ ] Multi-sample integration
- [ ] Perturbation predictions
- [ ] Report generation

### Phase 4: User Experience
- [ ] Web application deployment
- [ ] Jupyter notebook widgets
- [ ] API for programmatic access
- [ ] Documentation and tutorials

---

## 📦 Dependencies

```
# Core
- numpy
- pandas
- scipy
- scanpy
- anndata

# Visualization
- matplotlib
- seaborn
- plotly
- streamlit

# Analysis
- scikit-learn
- statsmodels
- umap-learn

# Metabolic modeling
- cobrapy
```

---

## 🧬 Background: FBA + GEMs + scRNA-seq

**Flux Balance Analysis (FBA)** predicts metabolic fluxes through a biochemical network by optimizing an objective function (e.g., biomass production) subject to stoichiometric constraints.

**Genome-Scale Metabolic Models (GEMs)** are comprehensive reconstructions of an organism's metabolism, containing thousands of reactions and metabolites.

**COMPASS** (Characterizing Cell states through metabolic Profiling of the Transcriptome) integrates scRNA-seq data with GEMs to infer metabolic activity at single-cell resolution.

This tool brings these concepts together in an accessible visualization platform.

---

## License

MIT License © 2025 Omics Pilot

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
