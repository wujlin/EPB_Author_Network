# EPB Author Collaboration Network Analysis

This repository contains the code used for the author network analysis in the
paper "Evolution of Environment and Planning B Collaborations and Topics Over
the Decades". The code constructs the EPB author collaboration network from the
cleaned publication CSV, extracts the largest connected component (LCC), detects
collaboration communities, and generates the figures, tables, and summary
metrics used by the manuscript's network analysis section.

The repository covers the network analysis workflow only. It does not include
the full data collection pipeline, the text corpus preparation workflow, or the
BERTopic topic modeling workflow.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   `-- README.md
|-- docs/
|   `-- network_analysis_overview.md
|-- outputs/
|   `-- README.md
|-- scripts/
|   `-- run_network_analysis.py
`-- src/
    `-- legacy_network_analysis.py
```

## Manuscript Workflow Map

| Manuscript analysis item | Code path | Main output |
| --- | --- | --- |
| Cleaned author data input | `data/output_final_with_source_revised_cleaned_authorfix.csv` | Input CSV, not stored in this repository |
| Author collaboration network construction | `scripts/run_network_analysis.py`, `RevisedCSVNetworkBuilder.build_collaboration_network()` | `outputs/network_analysis_run/metrics/network_analysis_summary.json` |
| Weighted edge definition and weighted degree analysis | `scripts/run_network_analysis.py`, `weighted_degree()`, `NetworkAnalyzer.analyze_weighted_degree_distribution_detailed()` | `figures/03-weighted-degree-analysis.png`, `tables/top_authors_by_weighted_degree.csv` |
| Full network and LCC structural diagnostics | `scripts/run_network_analysis.py`, `create_author_network_structural_plot()` | `figures/08-structural-analysis.png` |
| Small-world and path-length analysis on the LCC | `scripts/run_network_analysis.py`, `render_small_world_clean()` | `figures/14-small-world-largest-component.png`, summary JSON |
| Louvain community detection on the LCC | `src/legacy_network_analysis.py`, `NetworkStructuralAnalyzer.detect_communities()` | `figures/community-detection-results.json` |
| Top 10 community table | `scripts/run_network_analysis.py`, `export_community_representatives()` | `tables/community_representatives.csv` |
| Community temporal evolution by decade | `src/legacy_network_analysis.py`, `NetworkStructuralAnalyzer.analyze_community_temporal_evolution()` and `scripts/run_network_analysis.py`, `render_community_decade_evolution_clean()` | `figures/02-community_decade_evolution.png` |
| Main network and community visualizations | `scripts/run_network_analysis.py`, browser/SVG/static rendering helpers | `figures/01-epbnetwork.png`, `figures/05-community_evolution_network.png` |

The current manuscript uses two stable network terms:

- `author collaboration network`: the full weighted co-authorship network built
  from all cleaned EPB publication records.
- `largest connected component (LCC) of the author collaboration network`: the
  largest connected subgraph extracted directly from the author collaboration
  network. Path-based measures, Louvain community detection, and temporal
  community analysis use this LCC.

The front-facing workflow is implemented in `scripts/run_network_analysis.py`.
The file `src/legacy_network_analysis.py` contains inherited helper classes for
network analysis and plotting. Some internal helper names are preserved for
compatibility, but the manuscript workflow itself extracts the LCC directly from
the full author collaboration network.

## Network Construction

The input data are the standardized author names in `authors_full_final`. Each
author is represented as a node. For each multi-authored paper, the code connects
every pair of co-authors with an undirected edge.

For a paper `p` with `n_p` unique authors, each co-author pair on that paper
receives a fractional contribution:

```text
w_ij^(p) = 1 / n_p
```

If two authors collaborated on multiple EPB papers, their final edge weight is
the sum of their paper-level contributions:

```text
W_ij = sum over papers p containing both i and j of (1 / n_p)
```

Single-authored papers add author nodes and paper counts, but they do not add
collaboration edges.

## Input Data

The script expects a cleaned EPB CSV with at least these columns:

- `title`
- `year`
- `authors`
- `authors_full_final`
- `doi`

Optional quality-tracking columns are also read when present:

- `authors_source_final`
- `paper_needs_review`
- `paper_review_reason`

By default, place the input CSV at:

```text
data/output_final_with_source_revised_cleaned_authorfix.csv
```

The CSV is excluded from version control by `.gitignore`. To run the code with a
local data file elsewhere, pass the path with `--input-csv`.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

Playwright is used only for browser-rendered PNG export of selected HTML/SVG
visualizations.

## Run

Default run:

```bash
python scripts/run_network_analysis.py
```

Run with an explicit input file and output directory:

```bash
python scripts/run_network_analysis.py \
  --input-csv data/output_final_with_source_revised_cleaned_authorfix.csv \
  --run-dir outputs/network_analysis_run
```

For a faster manuscript-oriented run that avoids optional browser-rendered
ECharts and Cosmograph candidates:

```bash
python scripts/run_network_analysis.py \
  --input-csv data/output_final_with_source_revised_cleaned_authorfix.csv \
  --run-dir outputs/network_analysis_run \
  --disable-echarts-auto-render \
  --disable-cosmograph-candidate
```

## Main Outputs

The run directory contains:

```text
outputs/network_analysis_run/
|-- figures/
|   |-- 01-epbnetwork.png
|   |-- 02-community_decade_evolution.png
|   |-- 03-weighted-degree-analysis.png
|   |-- 04-weighted-comparison.png
|   |-- 05-community_evolution_network.png
|   |-- 08-structural-analysis.png
|   |-- 14-small-world-largest-component.png
|   |-- community-detection-results.json
|   `-- network_data.json
|-- metrics/
|   `-- network_analysis_summary.json
`-- tables/
    |-- community_representatives.csv
    `-- top_authors_by_weighted_degree.csv
```

Generated outputs are excluded from version control by default.

## Reproducibility Notes

- Community detection uses Louvain on the LCC with `weight="weight"` and random
  seed `42`.
- Path length and diameter are computed on the LCC.
- Degree-based author summaries are computed on the full author collaboration
  network unless the output description explicitly states that the LCC is used.
- The expected revised CSV for the manuscript contains 2,831 EPB papers covering
  1974 through March 2026.
