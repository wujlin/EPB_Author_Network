#!/usr/bin/env python3
"""Run the EPB author collaboration network analysis.

This packaged entry point expects the revised EPB CSV as input and writes all
figures, tables, and metrics to a run directory. It keeps the analysis logic
used for the manuscript while making the paths portable for GitHub.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
from html import escape as html_escape
import importlib.util
import io
import json
import os
import random
import re
import shutil
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_ROOT / "src"
DEFAULT_INPUT = PACKAGE_ROOT / "data" / "output_final_with_source_revised_cleaned_authorfix.csv"
DEFAULT_RUN_DIR = PACKAGE_ROOT / "outputs" / "network_analysis_run"
DEFAULT_ESSAY_FIGURES = PACKAGE_ROOT / "outputs" / "essay_figures"
DEFAULT_LATEST_FIGURES = PACKAGE_ROOT / "outputs" / "latest_network_figures"
COSMOGRAPH_PACKAGE_VERSION = "2.3.2"
LATEST_HTML_ALLOWLIST = {
    "01-network-hybrid-force-pack.html",
    "06-community-evolution-network.html",
}
NETWORK_RENDER_PALETTE = [
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#9333ea",
    "#0f766e",
    "#ea580c",
    "#475569",
    "#ca8a04",
    "#0284c7",
    "#be123c",
    "#4d7c0f",
    "#7e22ce",
    "#0d9488",
    "#b91c1c",
    "#1d4ed8",
    "#a16207",
    "#0369a1",
    "#c2410c",
    "#15803d",
    "#86198f",
    "#0e7490",
    "#9f1239",
    "#3f6212",
    "#6d28d9",
    "#155e75",
    "#b45309",
    "#334155",
]

sys.path.insert(0, str(SRC_DIR))


def load_original_network_classes():
    module_path = SRC_DIR / "legacy_network_analysis.py"
    spec = importlib.util.spec_from_file_location("epb_network_analysis_original", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original network module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.EPBNetworkBuilder, module.NetworkAnalyzer, module.NetworkStructuralAnalyzer


EPBNetworkBuilder, NetworkAnalyzer, NetworkStructuralAnalyzer = load_original_network_classes()


def run_silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class RevisedCSVNetworkBuilder(EPBNetworkBuilder):
    """Thin adapter from the revised final CSV to the original builder schema."""

    def __init__(self, csv_path: Path, author_col: str = "authors_full_final"):
        self.csv_path = Path(csv_path)
        self.author_col = author_col
        self.author_mapping_file = None
        self.data_directory = str(self.csv_path.parent)
        self.author_mapping: dict[str, list[str]] = {}
        self.name_to_canonical: dict[str, str] = {}
        self.papers_data = pd.DataFrame()
        self.collaboration_network = nx.Graph()

        print("Initializing revised CSV network builder.")
        self._load_papers_data()

    def _load_author_mapping(self):
        self.author_mapping = {}
        self.name_to_canonical = {}

    def _load_papers_data(self):
        print(f"Loading revised paper data: {self.csv_path}")
        raw = pd.read_csv(self.csv_path)

        required = {"title", "year", "authors", self.author_col}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"Missing required columns in revised CSV: {missing}")

        year_numeric = pd.to_numeric(raw["year"], errors="coerce")
        author_values = raw[self.author_col].fillna(raw["authors"]).astype(str).str.strip()

        papers = pd.DataFrame(
            {
                "Title": raw["title"].fillna("").astype(str),
                "Author": author_values,
                "Publication Year": year_numeric,
                "Decade": year_numeric.map(decade_label),
                "Source title": "Environment and Planning B",
                "DOI": raw.get("doi", pd.Series([""] * len(raw))).fillna("").astype(str),
                "paper_needs_review": raw.get(
                    "paper_needs_review", pd.Series([""] * len(raw))
                ).fillna(""),
                "paper_review_reason": raw.get(
                    "paper_review_reason", pd.Series([""] * len(raw))
                ).fillna(""),
                "authors_source_final": raw.get(
                    "authors_source_final", pd.Series([""] * len(raw))
                ).fillna(""),
            }
        )

        self.papers_data = papers
        print(f"Loaded {len(self.papers_data)} papers from revised CSV.")
        print(
            "   Year range:",
            int(year_numeric.min()) if year_numeric.notna().any() else "NA",
            "-",
            int(year_numeric.max()) if year_numeric.notna().any() else "NA",
        )
        print("   Decades:", ", ".join(sorted(papers["Decade"].dropna().unique())))

    def build_collaboration_network(self, use_weighted: bool = True) -> nx.Graph:
        """Build the collaboration network with deterministic author ordering.

        This mirrors the original EPBNetworkBuilder logic. The only intentional
        implementation difference is replacing list(set(authors)) with
        sorted(set(authors)) so node/edge insertion order is stable across
        Python processes while preserving the exact author set and weights.
        """
        from collections import defaultdict

        mode = "weighted" if use_weighted else "unweighted"
        print(f"\nBuilding author collaboration network ({mode}).")

        collaboration_weights = defaultdict(float)
        collaboration_counts = defaultdict(int)
        author_papers = defaultdict(list)

        for idx, row in self.papers_data.iterrows():
            if pd.isna(row.get("Author")):
                continue

            authors = self._split_authors(row["Author"])
            standardized_authors = [self._standardize_author_name(author) for author in authors]
            standardized_authors = sorted(set(standardized_authors))

            paper_info = {
                "title": row.get("Title", ""),
                "year": row.get("Publication Year", ""),
                "decade": row.get("Decade", ""),
                "journal": row.get("Source title", ""),
                "index": idx,
                "n_authors": len(standardized_authors),
            }

            for author in standardized_authors:
                author_papers[author].append(paper_info)

            if len(standardized_authors) > 1:
                paper_weight = 1.0 / len(standardized_authors) if use_weighted else 1.0
                for i in range(len(standardized_authors)):
                    for j in range(i + 1, len(standardized_authors)):
                        author1, author2 = standardized_authors[i], standardized_authors[j]
                        edge = tuple(sorted([author1, author2]))
                        collaboration_weights[edge] += paper_weight
                        collaboration_counts[edge] += 1

        self.collaboration_network = nx.Graph()

        for author, papers in author_papers.items():
            paper_types = {}
            for paper in papers:
                n_authors = paper["n_authors"]
                paper_types[n_authors] = paper_types.get(n_authors, 0) + 1

            self.collaboration_network.add_node(
                author,
                paper_count=len(papers),
                papers=papers,
                paper_types=paper_types,
            )

        total_weight = 0.0
        for edge, weight in collaboration_weights.items():
            author1, author2 = edge
            count = collaboration_counts[edge]
            self.collaboration_network.add_edge(
                author1,
                author2,
                weight=weight,
                count=count,
                avg_weight=weight / count,
            )
            total_weight += weight

        print("Collaboration network built.")
        print(f"  - Author nodes: {self.collaboration_network.number_of_nodes()}")
        print(f"  - Collaboration edges: {self.collaboration_network.number_of_edges()}")
        if use_weighted and self.collaboration_network.number_of_edges() > 0:
            print(f"  - Total edge weight: {total_weight:.2f}")
            print(
                "  - Mean edge weight: "
                f"{total_weight / self.collaboration_network.number_of_edges():.4f}"
            )

        return self.collaboration_network


def decade_label(year: Any) -> str:
    if pd.isna(year):
        return "unknown"
    year_int = int(float(year))
    if year_int < 1980:
        return "1970s"
    if year_int < 1990:
        return "1980s"
    if year_int < 2000:
        return "1990s"
    if year_int < 2010:
        return "2000s"
    if year_int < 2020:
        return "2010s"
    return "2020s"


def weighted_degree(graph: nx.Graph, node: str) -> float:
    return sum(data.get("weight", 1.0) for _, _, data in graph.edges(node, data=True))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def top_community_ids(communities_result: dict[str, Any], n: int = 10) -> list[int]:
    sizes = communities_result.get("community_sizes", {})
    return [
        int(cid)
        for cid, _ in sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:n]
    ]


def export_community_representatives(
    graph: nx.Graph,
    communities_result: dict[str, Any],
    output_csv: Path,
    top_n: int = 15,
    key_members: int = 4,
) -> pd.DataFrame:
    partition = communities_result["partition"]
    sizes = communities_result["community_sizes"]
    rows = []

    for rank, (comm_id, size) in enumerate(
        sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:top_n], start=1
    ):
        members = [node for node, cid in partition.items() if cid == comm_id]
        member_scores = sorted(
            (
                (
                    member,
                    weighted_degree(graph, member),
                    graph.nodes[member].get("paper_count", 0),
                    graph.degree(member),
                )
                for member in members
            ),
            key=lambda item: (item[1], item[2], item[3], item[0]),
            reverse=True,
        )
        top_star = member_scores[0][0] if member_scores else ""
        keys = [member for member, *_ in member_scores[1 : 1 + key_members]]
        rows.append(
            {
                "rank": rank,
                "community_id": comm_id,
                "size": size,
                "top_star": top_star,
                "top_star_weighted_degree": member_scores[0][1] if member_scores else 0,
                "key_members": "; ".join(keys),
            }
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Community representatives saved to: {output_csv}")
    return df


def export_top_authors(graph: nx.Graph, output_csv: Path, top_n: int = 50) -> pd.DataFrame:
    rows = []
    for author in graph.nodes:
        rows.append(
            {
                "author": author,
                "paper_count": graph.nodes[author].get("paper_count", 0),
                "weighted_degree": weighted_degree(graph, author),
                "unweighted_degree": graph.degree(author),
            }
        )
    df = pd.DataFrame(rows).sort_values(
        ["weighted_degree", "paper_count", "unweighted_degree", "author"],
        ascending=[False, False, False, True],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.head(top_n).to_csv(output_csv, index=False)
    print(f"Top authors saved to: {output_csv}")
    return df.head(top_n)


def largest_connected_component_subgraph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot extract LCC from an empty graph")
    largest_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_component).copy()


def degree_neighbor_correlation_data(graph: nx.Graph) -> dict[str, Any]:
    degrees = {}
    for node in graph.nodes():
        degrees[node] = weighted_degree(graph, node)

    neighbor_degrees = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if neighbors:
            neighbor_degrees[node] = float(np.mean([degrees[neighbor] for neighbor in neighbors]))
        else:
            neighbor_degrees[node] = 0.0

    degree_values = np.array([degrees[node] for node in graph.nodes()], dtype=float)
    neighbor_values = np.array([neighbor_degrees[node] for node in graph.nodes()], dtype=float)
    if len(degree_values) < 2 or np.std(degree_values) == 0 or np.std(neighbor_values) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(degree_values, neighbor_values)[0, 1])

    return {
        "correlation": correlation,
        "degrees": degree_values,
        "neighbor_degrees": neighbor_values,
    }


def clustering_by_degree_data(graph: nx.Graph) -> dict[str, Any]:
    clustering = nx.clustering(graph, weight="weight")
    average_clustering = float(nx.average_clustering(graph, weight="weight"))
    grouped: dict[int, list[float]] = {}
    for node in graph.nodes():
        degree_group = int(round(weighted_degree(graph, node)))
        grouped.setdefault(degree_group, []).append(float(clustering[node]))

    degrees = sorted(grouped)
    average_by_degree = [float(np.mean(grouped[degree])) for degree in degrees]
    return {
        "average_clustering": average_clustering,
        "degrees": degrees,
        "average_by_degree": average_by_degree,
        "group_sizes": [len(grouped[degree]) for degree in degrees],
    }


def apply_clean_axis_style(ax):
    ax.grid(True, color="#d7dee8", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(colors="#263241", labelsize=8.5)


def create_author_network_structural_plot(
    graph: nx.Graph,
    lcc_graph: nx.Graph,
    save_path: Path,
) -> dict[str, Any]:
    """Render structural diagnostics for the full network and its LCC."""
    print("Creating clean structural analysis plot for author network scopes.")
    scopes = [
        ("author_collaboration_network", "Author collaboration network", graph),
        (
            "lcc_author_collaboration_network",
            "LCC of author collaboration network",
            lcc_graph,
        ),
    ]
    results: dict[str, Any] = {}

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)

        for idx, (scope_key, scope_label, scope_graph) in enumerate(scopes):
            corr = degree_neighbor_correlation_data(scope_graph)
            cluster = clustering_by_degree_data(scope_graph)
            results[scope_key] = {
                "correlation": corr,
                "clustering": cluster,
            }

            degree_values = corr["degrees"]
            neighbor_values = corr["neighbor_degrees"]
            log_degrees = np.log1p(degree_values)
            log_neighbor_values = np.log1p(neighbor_values)

            ax = axes[0, idx]
            if len(degree_values) > 0:
                ax.hexbin(
                    log_degrees,
                    log_neighbor_values,
                    gridsize=36,
                    mincnt=1,
                    bins="log",
                    cmap="Blues",
                    linewidths=0,
                    alpha=0.92,
                )
                if len(log_degrees) > 2 and np.std(log_degrees) > 0:
                    z = np.polyfit(log_degrees, log_neighbor_values, 1)
                    x_trend = np.linspace(float(log_degrees.min()), float(log_degrees.max()), 100)
                    ax.plot(
                        x_trend,
                        np.poly1d(z)(x_trend),
                        "--",
                        color="#d94841",
                        linewidth=1.8,
                        alpha=0.9,
                    )
            ax.text(
                0.04,
                0.94,
                f"r = {corr['correlation']:.3f}\nn = {scope_graph.number_of_nodes():,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d7dee8"),
            )
            ax.set_title(f"{chr(97 + idx)}. {scope_label}: degree mixing", fontweight="bold")
            ax.set_xlabel("log(1 + weighted degree)")
            ax.set_ylabel("log(1 + avg. neighbor degree)" if idx == 0 else "")
            apply_clean_axis_style(ax)

            ax = axes[1, idx]
            if len(degree_values) > 1:
                clustering_values = nx.clustering(scope_graph, weight="weight")
                clustering_df = pd.DataFrame(
                    {
                        "log_degree": log_degrees,
                        "clustering": [
                            float(clustering_values[node]) for node in scope_graph.nodes()
                        ],
                    }
                )
                q = min(12, max(2, clustering_df["log_degree"].nunique()))
                clustering_df["degree_bin"] = pd.qcut(
                    clustering_df["log_degree"], q=q, duplicates="drop"
                )
                grouped = (
                    clustering_df.groupby("degree_bin", observed=True)
                    .agg(
                        center=("log_degree", "mean"),
                        clustering=("clustering", "mean"),
                        count=("clustering", "size"),
                    )
                    .reset_index(drop=True)
                )
                sizes = 40 + 360 * np.sqrt(grouped["count"] / grouped["count"].max())
                ax.scatter(
                    grouped["center"],
                    grouped["clustering"],
                    s=sizes,
                    color="#2a9d8f",
                    alpha=0.78,
                    edgecolors="white",
                    linewidth=0.9,
                )
                if len(grouped) > 2:
                    z = np.polyfit(grouped["center"], grouped["clustering"], 1)
                    x_trend = np.linspace(grouped["center"].min(), grouped["center"].max(), 100)
                    ax.plot(
                        x_trend,
                        np.poly1d(z)(x_trend),
                        "--",
                        color="#2f855a",
                        linewidth=1.8,
                        alpha=0.9,
                        label=f"slope = {z[0]:.3f}",
                    )
                average_clustering = cluster["average_clustering"]
                ax.axhline(
                    y=average_clustering,
                    color="#f59e0b",
                    linestyle="-",
                    linewidth=1.8,
                    alpha=0.82,
                    label=f"global avg. = {average_clustering:.3f}",
                )
                ax.set_ylim(bottom=0)
                ax.legend(frameon=True, framealpha=0.94, loc="upper right")
            ax.set_title(f"{chr(100 + idx)}. {scope_label}: clustering", fontweight="bold")
            ax.set_xlabel("log(1 + weighted degree)")
            ax.set_ylabel("avg. weighted clustering" if idx == 0 else "")
            apply_clean_axis_style(ax)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Clean structural analysis plot saved to: {save_path}")
    return results


def render_small_world_clean(graph: nx.Graph, output_path: Path) -> dict[str, Any]:
    lcc = largest_connected_component_subgraph(graph)
    distances = []
    for source, lengths in nx.all_pairs_shortest_path_length(lcc):
        distances.extend(distance for target, distance in lengths.items() if source != target)
    if not distances:
        raise ValueError("Cannot render small-world analysis without path lengths")

    max_distance = int(max(distances))
    average_path_length = float(np.mean(distances))
    distance_array = np.array(distances)
    coverage_by_distance = {
        d: float(np.mean(distance_array <= d)) for d in range(1, max_distance + 1)
    }
    steps_90 = next((d for d, coverage in coverage_by_distance.items() if coverage >= 0.9), None)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 180,
        }
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.7), constrained_layout=True)

        bins = np.arange(0.5, max_distance + 1.6, 1)
        counts, _, patches = ax1.hist(
            distances,
            bins=bins,
            color="#3b82bd",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.88,
        )
        max_count = max(counts) if len(counts) else 1
        for count, patch in zip(counts, patches):
            patch.set_facecolor(plt.cm.Blues(0.36 + 0.50 * (count / max_count)))
        ax1.axvline(
            average_path_length,
            color="#f59e0b",
            lw=2.0,
            ls="--",
            label=f"mean = {average_path_length:.2f}",
        )
        ax1.axvline(
            max_distance,
            color="#d95f45",
            lw=2.0,
            label=f"diameter = {max_distance}",
        )
        ax1.set_title("a. Path length distribution", weight="bold")
        ax1.set_xlabel("Shortest path length")
        ax1.set_ylabel("Ordered author-pair count")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.legend(loc="upper left", frameon=True, framealpha=0.94)
        apply_clean_axis_style(ax1)

        steps = np.array(list(coverage_by_distance.keys()))
        coverage = np.array(list(coverage_by_distance.values()))
        ax2.fill_between(steps, coverage, color="#b56591", alpha=0.22)
        ax2.plot(
            steps,
            coverage,
            color="#a63d78",
            lw=2.2,
            marker="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
        ax2.axhline(0.9, color="#d95f45", lw=1.8, ls="--", label="90% coverage")
        if steps_90 is not None:
            ax2.axvline(steps_90, color="#64748b", lw=1.3, ls=":", alpha=0.8)
            ax2.text(
                steps_90 + 0.15,
                0.08,
                f"90% at {steps_90} steps",
                fontsize=8.5,
                color="#334155",
            )
        ax2.set_title("b. Reachability by path length", weight="bold")
        ax2.set_xlabel("Maximum number of steps")
        ax2.set_ylabel("Coverage ratio")
        ax2.set_ylim(0, 1.04)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(loc="upper left", frameon=True, framealpha=0.94)
        apply_clean_axis_style(ax2)

        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"Clean small-world analysis plot saved to: {output_path}")
    return {
        "diameter": max_distance,
        "average_path_length": average_path_length,
        "is_small_world": average_path_length < 6,
        "steps_for_90_percent_coverage": steps_90,
    }


def render_community_decade_evolution_clean(
    graph: nx.Graph,
    temporal_results: dict[str, Any],
    output_path: Path,
) -> None:
    decades = ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    communities = list(temporal_results.get("target_communities", []))
    partition = temporal_results.get("current_communities", {}).get("partition", {})
    weighted_degrees = {node: weighted_degree(graph, node) for node in graph.nodes()}

    labels = []
    for community_id in communities:
        members = [node for node, cid in partition.items() if cid == community_id and node in graph]
        if members:
            leader = max(
                members,
                key=lambda node: (
                    weighted_degrees.get(node, 0.0),
                    graph.nodes[node].get("paper_count", 0),
                    node,
                ),
            )
            labels.append(f"C{community_id}: {display_surname(leader)}")
        else:
            labels.append(f"C{community_id}")

    rows = []
    for community_index, community_id in enumerate(communities):
        for decade_index, decade in enumerate(decades):
            community_data = (
                temporal_results.get("decade_analysis", {})
                .get(decade, {})
                .get("communities", {})
                .get(community_id, {})
            )
            rows.append(
                {
                    "x": decade_index,
                    "y": community_index,
                    "active_members": community_data.get("active_members", 0),
                    "papers": community_data.get("papers_count", 0),
                }
            )
    df = pd.DataFrame(rows)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 9.5,
            "figure.dpi": 180,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7), constrained_layout=True)
        panel_specs = [
            ("active_members", "a. Active members over time", "Active members", "Oranges"),
            ("papers", "b. Papers published over time", "Papers published", "Blues"),
        ]
        for ax, (column, title, colorbar_label, cmap) in zip(axes, panel_specs):
            plot_df = df[df[column] > 0].copy()
            max_value = max(float(df[column].max()), 1.0)
            if len(plot_df):
                sizes = 42 + 640 * np.sqrt(plot_df[column] / max_value)
                scatter = ax.scatter(
                    plot_df["x"],
                    plot_df["y"],
                    s=sizes,
                    c=plot_df[column],
                    cmap=cmap,
                    vmin=max(1.0, float(plot_df[column].min())),
                    vmax=max_value,
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.9,
                )
                colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.018)
                colorbar.set_label(colorbar_label, fontsize=9)
                colorbar.ax.tick_params(labelsize=8)
            ax.set_title(title, weight="bold")
            ax.set_xticks(range(len(decades)))
            ax.set_xticklabels(decades)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Decade")
            ax.set_xlim(-0.55, len(decades) - 0.45)
            ax.set_ylim(len(labels) - 0.55, -0.55)
            apply_clean_axis_style(ax)

        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"Clean community-decade evolution plot saved to: {output_path}")


def render_static_network_png(
    graph: nx.Graph,
    communities_result: dict[str, Any] | None,
    output_path: Path,
    max_nodes: int = 900,
    seed: int = 42,
):
    """Render a stable manuscript-oriented overview network figure."""
    print(f"Rendering static network PNG: {output_path}")
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot render empty network")

    lcc = max(nx.connected_components(graph), key=len)
    lcc_graph = graph.subgraph(lcc).copy()
    ranked = rank_nodes_for_render(lcc_graph)
    selected = ranked[:max_nodes]
    subgraph = lcc_graph.subgraph(selected).copy()
    subgraph.remove_nodes_from([n for n in subgraph.nodes if subgraph.degree(n) == 0])
    if subgraph.number_of_nodes() == 0:
        raise ValueError("Cannot render network after node selection")

    pos = nx.spring_layout(subgraph, seed=seed, weight="weight", iterations=120, k=0.42)
    pos = orient_layout_horizontally(pos)
    pos = stretch_layout_to_frame(pos, blend=0.88, aspect_ratio=16 / 9)
    partition = communities_result.get("partition", {}) if communities_result else {}

    nodes = list(subgraph.nodes())
    node_scores = np.array([weighted_degree(subgraph, node) for node in nodes], dtype=float)
    max_score = max(float(node_scores.max()), 1e-9)
    x_values = np.array([pos[node][0] for node in nodes], dtype=float)
    y_values = np.array([pos[node][1] for node in nodes], dtype=float)

    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    x_limits = (x_min - 0.035 * x_span, x_max + 0.035 * x_span)
    y_limits = (y_min - 0.055 * y_span, y_max + 0.055 * y_span)

    figure_size = (14.0, 8.0) if subgraph.number_of_nodes() > 500 else (10.5, 6.5)
    fig, ax = plt.subplots(figsize=figure_size, facecolor="white")

    edge_segments = []
    edge_colors = []
    edge_widths = []
    for source, target, data in subgraph.edges(data=True):
        edge_segments.append([(pos[source][0], pos[source][1]), (pos[target][0], pos[target][1])])
        community_id = int(partition.get(source, 0))
        edge_colors.append(to_rgba(NETWORK_RENDER_PALETTE[community_id % len(NETWORK_RENDER_PALETTE)], 0.085))
        edge_widths.append(0.12 + 0.35 * min(float(data.get("weight", 1.0)), 2.5) / 2.5)
    ax.add_collection(
        LineCollection(
            edge_segments,
            colors=edge_colors,
            linewidths=edge_widths,
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
    )

    node_sizes = 5.0 + 58.0 * np.power(np.maximum(node_scores, 0) / max_score, 0.70)
    node_colors = [
        NETWORK_RENDER_PALETTE[int(partition.get(node, 0)) % len(NETWORK_RENDER_PALETTE)]
        for node in nodes
    ]
    ax.scatter(
        x_values,
        y_values,
        s=node_sizes,
        c=node_colors,
        edgecolors="white",
        linewidths=0.32,
        alpha=0.9,
        zorder=2,
    )

    ranked_subgraph_nodes = [node for node in rank_nodes_for_render(subgraph) if node in subgraph]
    label_limit = 46 if subgraph.number_of_nodes() > 500 else min(38, subgraph.number_of_nodes())
    label_candidates = ranked_subgraph_nodes[: max(label_limit * 3, label_limit)]
    label_placements = compute_label_placements(
        subgraph,
        pos,
        label_candidates,
        x_limits,
        y_limits,
        figure_size,
        label_limit,
        max_score,
    )
    for node, x, y, horizontal_align, vertical_align, font_size, label in label_placements:
        ax.text(
            x,
            y,
            label,
            fontsize=font_size,
            fontweight="bold",
            color="#1f2937",
            ha=horizontal_align,
            va=vertical_align,
            zorder=3,
            path_effects=[
                path_effects.withStroke(linewidth=1.45, foreground="white", alpha=0.97)
            ],
        )

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=420, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def rank_nodes_for_render(graph: nx.Graph) -> list[str]:
    return sorted(
        graph.nodes(),
        key=lambda node: (
            -weighted_degree(graph, node),
            -graph.nodes[node].get("paper_count", 0),
            -graph.degree(node),
            str(node),
        ),
    )


def orient_layout_horizontally(pos: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    nodes = list(pos)
    coords = np.array([pos[node] for node in nodes], dtype=float)
    if coords.shape[0] < 3:
        return {node: (float(coords[idx, 0]), float(coords[idx, 1])) for idx, node in enumerate(nodes)}

    coords = coords - coords.mean(axis=0)
    covariance = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    rotation = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    rotated = coords @ rotation
    return {node: (float(rotated[idx, 0]), float(rotated[idx, 1])) for idx, node in enumerate(nodes)}


def stretch_layout_to_frame(
    pos: dict[str, tuple[float, float]],
    blend: float = 0.88,
    aspect_ratio: float = 16 / 9,
    margin: float = 0.025,
) -> dict[str, tuple[float, float]]:
    nodes = list(pos)
    coords = np.array([pos[node] for node in nodes], dtype=float)
    stretched = np.zeros_like(coords)

    for dimension in [0, 1]:
        values = coords[:, dimension]
        value_span = max(float(np.ptp(values)), 1e-12)
        normalized = (values - float(values.min())) / value_span
        order = np.argsort(values)
        ranks = np.empty(len(nodes), dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(nodes))
        stretched[:, dimension] = (1 - blend) * normalized + blend * ranks

    stretched[:, 0] = (stretched[:, 0] - 0.5) * aspect_ratio
    stretched[:, 1] = stretched[:, 1] - 0.5
    stretched[:, 0] *= 1 - 2 * margin
    stretched[:, 1] *= 1 - 2 * margin

    return {
        node: (float(stretched[idx, 0]), float(stretched[idx, 1]))
        for idx, node in enumerate(nodes)
    }


def compute_label_placements(
    graph: nx.Graph,
    pos: dict[str, tuple[float, float]],
    candidate_nodes: list[str],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    figure_size: tuple[float, float],
    max_labels: int,
    max_score: float,
) -> list[tuple[str, float, float, str, str, float, str]]:
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    boxes: list[tuple[float, float, float, float]] = []
    placements: list[tuple[str, float, float, str, str, float, str]] = []
    label_slots = [
        (1, 0, "left", "center"),
        (-1, 0, "right", "center"),
        (0, 1, "center", "bottom"),
        (0, -1, "center", "top"),
        (1, 1, "left", "bottom"),
        (-1, 1, "right", "bottom"),
        (1, -1, "left", "top"),
        (-1, -1, "right", "top"),
    ]

    for node in candidate_nodes:
        if node not in graph:
            continue
        label = display_surname(str(node))
        score_ratio = min(weighted_degree(graph, node) / max(max_score, 1e-9), 1.0)
        font_size = 5.2 + 2.2 * score_ratio**0.55
        label_width = (len(label) * font_size * 0.56 / 72.0) / figure_size[0] * x_span
        label_height = (font_size * 1.45 / 72.0) / figure_size[1] * y_span
        node_x, node_y = pos[node]
        offset_x = 0.009 * x_span + 0.003 * x_span * score_ratio
        offset_y = 0.014 * y_span + 0.004 * y_span * score_ratio

        for dx, dy, horizontal_align, vertical_align in label_slots:
            label_x = node_x + dx * offset_x
            label_y = node_y + dy * offset_y
            box = label_bounds(
                label_x,
                label_y,
                label_width,
                label_height,
                horizontal_align,
                vertical_align,
            )
            if (
                box[0] < x_limits[0]
                or box[2] > x_limits[1]
                or box[1] < y_limits[0]
                or box[3] > y_limits[1]
            ):
                continue
            if any(label_boxes_overlap(box, existing) for existing in boxes):
                continue
            boxes.append(box)
            placements.append(
                (node, label_x, label_y, horizontal_align, vertical_align, font_size, label)
            )
            break

        if len(placements) >= max_labels:
            break

    return placements


def label_bounds(
    x: float,
    y: float,
    width: float,
    height: float,
    horizontal_align: str,
    vertical_align: str,
) -> tuple[float, float, float, float]:
    if horizontal_align == "left":
        x_min, x_max = x, x + width
    elif horizontal_align == "right":
        x_min, x_max = x - width, x
    else:
        x_min, x_max = x - width / 2, x + width / 2

    if vertical_align == "bottom":
        y_min, y_max = y, y + height
    elif vertical_align == "top":
        y_min, y_max = y - height, y
    else:
        y_min, y_max = y - height / 2, y + height / 2

    return (x_min, y_min, x_max, y_max)


def label_boxes_overlap(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    padding: float = 0.006,
) -> bool:
    return not (
        box_a[2] + padding < box_b[0]
        or box_a[0] - padding > box_b[2]
        or box_a[3] + padding < box_b[1]
        or box_a[1] - padding > box_b[3]
    )


def display_surname(name: str) -> str:
    if "," in name:
        return name.split(",", 1)[0].strip()
    parts = name.split()
    return parts[-1] if len(parts) > 1 else name


def screenshot_html_element(
    html_file: Path,
    selector: str,
    output_path: Path,
    viewport: tuple[int, int],
    wait_ms: int = 14000,
) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(f"Playwright unavailable; cannot screenshot {html_file.name}: {exc}")
        return False

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handler = partial(QuietHandler, directory=str(html_file.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/{html_file.name}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": viewport[0], "height": viewport[1]},
                device_scale_factor=1,
            )
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(wait_ms)
            locator = page.locator(selector)
            locator.wait_for(state="visible", timeout=30000)
            locator.screenshot(path=str(output_path))
            browser.close()
        print(f"Screenshot saved: {output_path}")
        return True
    except Exception as exc:
        print(f"Failed to screenshot {html_file.name}: {exc}")
        return False
    finally:
        server.shutdown()
        server.server_close()


def render_echarts_force_png(
    graph: nx.Graph,
    communities_result: dict[str, Any],
    output_path: Path,
    figures_dir: Path,
    max_components: int = 30,
    max_nodes: int = 2200,
    label_count: int = -1,
    settle_ms: int = 24000,
    viewport: tuple[int, int] = (4200, 2400),
    pixel_ratio: int = 2,
) -> bool:
    """Render the overview network through an automated ECharts force layout."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(f"Playwright unavailable; cannot render ECharts PNG: {exc}")
        return False

    try:
        render_graph = select_echarts_render_graph(
            graph,
            max_components=max_components,
            max_nodes=max_nodes,
        )
        if render_graph.number_of_nodes() == 0:
            raise ValueError("Cannot render an empty ECharts network")

        data_path = figures_dir / "01-network-echarts-auto-data.json"
        html_path = figures_dir / "01-network-echarts-auto-render.html"
        render_data = build_echarts_force_render_data(
            render_graph,
            communities_result,
            label_count=label_count,
        )
        with data_path.open("w", encoding="utf-8") as f:
            json.dump(render_data, f, ensure_ascii=False, indent=2)
        write_echarts_force_render_html(
            html_path=html_path,
            data_file=data_path.name,
            viewport=viewport,
            pixel_ratio=pixel_ratio,
            settle_ms=settle_ms,
        )

        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                return

        handler = partial(QuietHandler, directory=str(figures_dir))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/{html_path.name}"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={"width": viewport[0], "height": viewport[1]},
                    device_scale_factor=1,
                )
                page.goto(url, wait_until="networkidle", timeout=60000)
                page.wait_for_function(
                    "window.__done === true",
                    timeout=max(45000, settle_ms + 20000),
                )
                data_url = page.evaluate("window.__exported")
                browser.close()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
            print(
                "ECharts PNG rendered: "
                f"{output_path} ({render_graph.number_of_nodes()} nodes, "
                f"{render_graph.number_of_edges()} edges)"
            )
            return True
        finally:
            server.shutdown()
            server.server_close()
    except Exception as exc:
        print(f"ECharts automated render failed; falling back to static PNG: {exc}")
        return False


def select_echarts_render_graph(
    graph: nx.Graph,
    max_components: int = 30,
    max_nodes: int = 2200,
) -> nx.Graph:
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    selected_nodes: list[str] = []
    for component in components[:max_components]:
        if selected_nodes and len(selected_nodes) + len(component) > max_nodes:
            break
        selected_nodes.extend(component)

    selected = graph.subgraph(selected_nodes).copy()
    if selected.number_of_nodes() > max_nodes:
        ranked = rank_nodes_for_render(selected)
        selected = selected.subgraph(ranked[:max_nodes]).copy()
    selected.remove_nodes_from([node for node in selected.nodes if selected.degree(node) == 0])
    return selected


def build_echarts_force_render_data(
    graph: nx.Graph,
    communities_result: dict[str, Any],
    label_count: int = -1,
) -> dict[str, Any]:
    partition = communities_result.get("partition", {}) if communities_result else {}
    ranked_nodes = rank_nodes_for_render(graph)
    if label_count <= 0 or label_count >= graph.number_of_nodes():
        label_nodes = set(graph.nodes())
    else:
        label_nodes = set(ranked_nodes[:label_count])
    max_score = max((weighted_degree(graph, node) for node in graph.nodes()), default=1.0)
    node_to_id = {node: str(index) for index, node in enumerate(graph.nodes())}

    nodes = []
    for node, node_id in node_to_id.items():
        score = weighted_degree(graph, node)
        score_ratio = score / max(max_score, 1e-9)
        community_id = partition.get(node)
        color = (
            NETWORK_RENDER_PALETTE[int(community_id) % len(NETWORK_RENDER_PALETTE)]
            if community_id is not None
            else "#94a3b8"
        )
        nodes.append(
            {
                "id": node_id,
                "name": display_surname(str(node)),
                "full_name": str(node),
                "value": round(score, 3),
                "papers": graph.nodes[node].get("paper_count", 0),
                "symbolSize": round(8.0 + 30.0 * score_ratio**0.58, 2),
                "itemStyle": {
                    "color": color,
                    "borderColor": "#ffffff",
                    "borderWidth": 0.65,
                },
                "label": {
                    "show": node in label_nodes,
                    "fontSize": round(13.0 + 5.5 * score_ratio**0.5, 1),
                    "fontWeight": "bold",
                    "color": "#1f2937",
                },
            }
        )

    links = []
    for source, target, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        links.append(
            {
                "source": node_to_id[source],
                "target": node_to_id[target],
                "value": round(weight, 3),
                "lineStyle": {
                    "width": round(0.2 + 0.72 * min(weight, 2.5) / 2.5, 3),
                    "opacity": 0.105,
                },
            }
        )

    return {
        "nodes": nodes,
        "links": links,
        "network_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "components": nx.number_connected_components(graph),
            "labeled_nodes": len(label_nodes),
        },
    }


def write_echarts_force_render_html(
    html_path: Path,
    data_file: str,
    viewport: tuple[int, int],
    pixel_ratio: int,
    settle_ms: int,
):
    width, height = viewport
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EPB Network Render</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
  <style>
    html, body, #chart {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: #ffffff;
      overflow: hidden;
    }}
  </style>
</head>
<body>
  <div id="chart"></div>
  <script>
    const WIDTH = {width};
    const HEIGHT = {height};
    const DATA_FILE = "{data_file}";
    const SETTLE_MS = {settle_ms};
    const PIXEL_RATIO = {pixel_ratio};
    const chart = echarts.init(
      document.getElementById("chart"),
      null,
      {{ renderer: "canvas", devicePixelRatio: 1 }}
    );

    async function main() {{
      const response = await fetch(DATA_FILE);
      const networkData = await response.json();
      chart.setOption({{
        backgroundColor: "#ffffff",
        series: [{{
          type: "graph",
          layout: "force",
          data: networkData.nodes,
          links: networkData.links,
          left: 0,
          top: 0,
          right: 0,
          bottom: 0,
          roam: false,
          draggable: false,
          animation: false,
          focusNodeAdjacency: false,
          label: {{
            show: true,
            formatter: "{{b}}",
            position: "right",
            distance: 4,
            textBorderColor: "#ffffff",
            textBorderWidth: 3
          }},
          labelLayout: {{
            hideOverlap: true,
            moveOverlap: "shiftY"
          }},
          force: {{
            repulsion: [180, 760],
            edgeLength: [85, 280],
            gravity: 0.03,
            friction: 0.7,
            layoutAnimation: true
          }},
          lineStyle: {{
            color: "source",
            opacity: 0.105,
            curveness: 0.03
          }}
        }}]
      }});

      setTimeout(() => finalize(networkData), SETTLE_MS);
    }}

    function finalize(networkData) {{
      const series = chart.getModel().getSeriesByIndex(0);
      const data = series.getData();
      const layouts = [];
      for (let i = 0; i < data.count(); i++) {{
        const point = data.getItemLayout(i);
        layouts.push([
          point && isFinite(point[0]) ? point[0] : 0,
          point && isFinite(point[1]) ? point[1] : 0
        ]);
      }}

      const xs = layouts.map(point => point[0]);
      const ys = layouts.map(point => point[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const margin = 52;
      const scale = Math.min(
        (WIDTH - 2 * margin) / Math.max(maxX - minX, 1),
        (HEIGHT - 2 * margin) / Math.max(maxY - minY, 1)
      );
      const usedWidth = (maxX - minX) * scale;
      const usedHeight = (maxY - minY) * scale;
      const offsetX = (WIDTH - usedWidth) / 2;
      const offsetY = (HEIGHT - usedHeight) / 2;

      networkData.nodes.forEach((node, index) => {{
        node.x = offsetX + (layouts[index][0] - minX) * scale;
        node.y = offsetY + (layouts[index][1] - minY) * scale;
        node.fixed = true;
      }});

      chart.clear();
      chart.setOption({{
        backgroundColor: "#ffffff",
        series: [{{
          type: "graph",
          layout: "none",
          data: networkData.nodes,
          links: networkData.links,
          left: 0,
          top: 0,
          right: 0,
          bottom: 0,
          roam: false,
          draggable: false,
          animation: false,
          focusNodeAdjacency: false,
          label: {{
            show: true,
            formatter: "{{b}}",
            position: "right",
            distance: 4,
            textBorderColor: "#ffffff",
            textBorderWidth: 3
          }},
          labelLayout: {{
            hideOverlap: true,
            moveOverlap: "shiftY"
          }},
          lineStyle: {{
            color: "source",
            opacity: 0.105,
            curveness: 0.03
          }}
        }}]
      }});

      setTimeout(() => {{
        window.__exported = chart.getDataURL({{
          type: "png",
          pixelRatio: PIXEL_RATIO,
          backgroundColor: "#ffffff"
        }});
        window.__done = true;
      }}, 1200);
    }}

    main().catch(error => {{
      window.__error = String(error);
      window.__done = false;
    }});
  </script>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def build_cosmograph_render_data(
    graph: nx.Graph,
    communities_result: dict[str, Any],
) -> dict[str, Any]:
    partition = communities_result.get("partition", {}) if communities_result else {}
    node_to_id = {node: str(index) for index, node in enumerate(graph.nodes())}

    points = []
    for node, node_id in node_to_id.items():
        community_id = partition.get(node)
        score = weighted_degree(graph, node)
        points.append(
            {
                "id": node_id,
                "label": display_surname(str(node)),
                "full_name": str(node),
                "weight": round(max(score, 0.01), 4),
                "papers": int(graph.nodes[node].get("paper_count", 0)),
                "community": str(community_id) if community_id is not None else "unassigned",
            }
        )

    links = []
    for source, target, data in graph.edges(data=True):
        links.append(
            {
                "source": node_to_id[source],
                "target": node_to_id[target],
                "weight": round(float(data.get("weight", 1.0)), 4),
            }
        )

    return {
        "points": points,
        "links": links,
        "network_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "components": nx.number_connected_components(graph),
        },
    }


def write_cosmograph_render_html(
    html_path: Path,
    data_file: str,
    label_count: int,
    wait_ms: int,
):
    palette = json.dumps(NETWORK_RENDER_PALETTE[:12])
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EPB Cosmograph Candidate</title>
  <style>
    html, body, #cosmograph-container {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: #ffffff;
      overflow: hidden;
    }}
  </style>
</head>
<body>
  <div id="cosmograph-container"></div>
  <script type="module">
    import {{ Cosmograph, prepareCosmographData }} from "https://esm.sh/@cosmograph/cosmograph@{COSMOGRAPH_PACKAGE_VERSION}";

    const raw = await fetch("{data_file}").then(response => response.json());
    const dataConfig = {{
      points: {{
        pointIdBy: "id",
        pointLabelBy: "label",
        pointSizeBy: "weight",
        pointColorBy: "community"
      }},
      links: {{
        linkSourceBy: "source",
        linkTargetsBy: ["target"],
        linkWidthBy: "weight"
      }}
    }};
    const result = await prepareCosmographData(dataConfig, raw.points, raw.links);
    const container = document.getElementById("cosmograph-container");
    const cosmograph = new Cosmograph(container, {{
      points: result.points,
      links: result.links,
      ...result.cosmographConfig,
      backgroundColor: "#ffffff",
      pointColorPalette: {palette},
      linkColor: "#7b93a6",
      pointSizeStrategy: "auto",
      pointSizeRange: [3, 11],
      pointSizeScale: 1,
      linkWidthScale: 0.18,
      showLabels: true,
      showTopLabels: true,
      showTopLabelsLimit: {label_count},
      showDynamicLabels: false,
      pointLabelFontSize: 14,
      pointLabelColor: "#111827",
      labelMargin: 5,
      renderLinks: true,
      fitViewOnInit: true,
      fitViewDelay: 3200,
      fitViewPadding: 0.03,
      simulationDecay: 9000,
      simulationGravity: 0.16,
      simulationRepulsion: 0.75,
      simulationLinkDistance: 18,
      simulationLinkSpring: 1,
      simulationFriction: 0.92,
      randomSeed: 42,
      pixelRatio: 2,
      scalePointsOnZoom: false
    }});
    window.cosmograph = cosmograph;
    setTimeout(() => {{ window.__done = true; }}, {wait_ms});
  </script>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def render_cosmograph_candidate_png(
    graph: nx.Graph,
    communities_result: dict[str, Any],
    figures_dir: Path,
    max_components: int = 1,
    max_nodes: int = 500,
    label_count: int = 75,
    viewport: tuple[int, int] = (3200, 1900),
    wait_ms: int = 20000,
) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(f"Playwright unavailable; cannot render Cosmograph candidate: {exc}")
        return False

    try:
        render_graph = select_echarts_render_graph(
            graph,
            max_components=max_components,
            max_nodes=max_nodes,
        )
        if render_graph.number_of_nodes() == 0:
            raise ValueError("Cannot render an empty Cosmograph network")

        data_path = figures_dir / "01-epbnetwork_cosmograph_data.json"
        html_path = figures_dir / "01-epbnetwork_cosmograph_candidate.html"
        png_path = figures_dir / "01-epbnetwork_cosmograph_candidate.png"
        render_data = build_cosmograph_render_data(render_graph, communities_result)
        with data_path.open("w", encoding="utf-8") as f:
            json.dump(render_data, f, ensure_ascii=False, indent=2)
        write_cosmograph_render_html(
            html_path=html_path,
            data_file=data_path.name,
            label_count=label_count,
            wait_ms=wait_ms,
        )

        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                return

        handler = partial(QuietHandler, directory=str(figures_dir))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/{html_path.name}"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"],
                )
                page = browser.new_page(
                    viewport={"width": viewport[0], "height": viewport[1]},
                    device_scale_factor=1,
                )
                page.goto(url, wait_until="networkidle", timeout=90000)
                page.wait_for_function(
                    "window.__done === true",
                    timeout=max(60000, wait_ms + 50000),
                )
                png_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(png_path), full_page=False)
                browser.close()
            print(
                "Cosmograph candidate rendered: "
                f"{png_path} ({render_graph.number_of_nodes()} nodes, "
                f"{render_graph.number_of_edges()} edges)"
            )
            return True
        finally:
            server.shutdown()
            server.server_close()
    except Exception as exc:
        print(f"Cosmograph candidate render failed: {exc}")
        return False


def verify_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        small = rgb.resize((min(400, rgb.width), min(400, rgb.height)))
        arr = np.asarray(small)
        nonwhite_ratio = float(np.mean(np.any(arr < 245, axis=2)))
        return {
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "nonwhite_ratio": nonwhite_ratio,
            "file_size": path.stat().st_size,
        }


def copy_essay_figures(figures_dir: Path, essay_figures_dir: Path):
    mapping = {
        "01-epbnetwork.png": "01-epbnetwork.png",
        "02-community_decade_evolution.png": "02-community_decade_evolution.png",
        "03-weighted-degree-analysis.png": "03-weighted_degree_analysis.png",
        "04-weighted-comparison.png": "04-weighted_comparison.png",
        "05-community_evolution_network.png": "05-community_evolution_network.png",
        "08-structural-analysis.png": "08-structural_analysis.png",
        "14-small-world-largest-component.png": "09-small_world.png",
    }
    essay_figures_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src_name, dst_name in mapping.items():
        src = figures_dir / src_name
        dst = essay_figures_dir / dst_name
        if not src.exists():
            raise FileNotFoundError(f"Expected figure was not generated: {src}")
        shutil.copy2(src, dst)
        copied.append({"source": str(src), "target": str(dst)})
        print(f"Updated essay figure: {dst.name}")
    return copied


def copy_latest_figure_bundle(figures_dir: Path, latest_dir: Path, run_dir: Path) -> dict[str, Any]:
    latest_dir.mkdir(parents=True, exist_ok=True)
    allowed_suffixes = {".png", ".html", ".json"}

    for stale in sorted(latest_dir.iterdir()):
        if stale.is_file() and stale.suffix.lower() in allowed_suffixes:
            stale.unlink()

    copied = []
    for src in sorted(figures_dir.iterdir()):
        if not src.is_file() or src.suffix.lower() not in allowed_suffixes:
            continue
        if src.suffix.lower() == ".html" and src.name not in LATEST_HTML_ALLOWLIST:
            continue
        dst = latest_dir / src.name
        shutil.copy2(src, dst)
        copied.append(
            {
                "source": str(src),
                "target": str(dst),
                "bytes": dst.stat().st_size,
            }
        )

    manifest = {
        "purpose": "Centralized latest EPB network figure bundle for review and manual selection.",
        "run_dir": str(run_dir),
        "source_figures_dir": str(figures_dir),
        "latest_figures_dir": str(latest_dir),
        "copied_files": copied,
        "notes": [
            "The essay figure directory is updated only when --update-essay is used.",
            "The Cosmograph candidate is kept for comparison and is not copied over the essay main figure.",
        ],
    }
    manifest_path = latest_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(manifest), f, ensure_ascii=False, indent=2)

    readme_path = latest_dir / "README.md"
    readme_text = (
        "# Latest EPB Network Figures\n\n"
        "This directory is the centralized review bundle for the latest network-analysis run.\n\n"
        "Key files:\n"
        "- `01-epbnetwork.png`: current automated or fallback main-network candidate.\n"
        "- `01-network-hybrid-force-pack.html`: current browser workflow using SVG circles, D3 hybrid force, community anchors, collision, and viewport boundary forces.\n"
        "- `01-epbnetwork_cosmograph_candidate.png`: optional Cosmograph comparison candidate when generated.\n"
        "- `06-community-evolution-network.html`: supporting community-evolution interaction file.\n"
        "- `manifest.json`: source and target paths for the copied files.\n\n"
        "The essay figure directory is separate. It is updated only when the script is run with `--update-essay`.\n"
    )
    readme_path.write_text(readme_text, encoding="utf-8")

    print(f"Latest figure bundle updated: {latest_dir}")
    return {
        "latest_figures_dir": str(latest_dir),
        "manifest": str(manifest_path),
        "readme": str(readme_path),
        "copied_file_count": len(copied),
    }


def cleanup_obsolete_html(figures_dir: Path) -> list[str]:
    removed = []
    for path in sorted(figures_dir.glob("*.html")):
        if path.name in LATEST_HTML_ALLOWLIST:
            continue
        path.unlink()
        removed.append(str(path))
    if removed:
        print(f"Removed obsolete HTML files: {len(removed)}")
    return removed


def write_manual_echarts_network_html(html_path: Path, data_file: str):
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EPB Author Collaboration Network</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #ffffff;
      color: #1f2937;
      font-family: Arial, Helvetica, sans-serif;
    }
    .toolbar {
      height: 52px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      box-sizing: border-box;
      border-bottom: 1px solid #e5e7eb;
      background: #ffffff;
      white-space: nowrap;
      overflow-x: auto;
    }
    button, select, input {
      font: inherit;
    }
    button {
      height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #f8fafc;
      color: #111827;
      padding: 0 10px;
      cursor: pointer;
    }
    button:hover {
      background: #eef2f7;
    }
    button.active {
      background: #1f2937;
      border-color: #1f2937;
      color: #ffffff;
    }
    select {
      height: 34px;
      min-width: 150px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: #111827;
      padding: 0 8px;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #475569;
      font-size: 13px;
    }
    input[type="range"] {
      width: 96px;
    }
    #network-chart {
      width: 100vw;
      height: calc(100vh - 52px);
      background: #ffffff;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="lockBtn" onclick="toggleManualLayout()">Lock Layout</button>
    <button onclick="resetView()">Reset View</button>
    <button onclick="toggleLabels()">Toggle Labels</button>
    <button onclick="saveAsImage()">Save PNG</button>
    <select id="communitySelect" onchange="updateSelectionInfo()"></select>
    <button onclick="moveSelectedCommunity('top-left')">Top Left</button>
    <button onclick="moveSelectedCommunity('top-right')">Top Right</button>
    <button onclick="moveSelectedCommunity('bottom-left')">Bottom Left</button>
    <button onclick="moveSelectedCommunity('bottom-right')">Bottom Right</button>
    <button onclick="nudgeSelectedCommunity(-1, 0)">Left</button>
    <button onclick="nudgeSelectedCommunity(1, 0)">Right</button>
    <button onclick="nudgeSelectedCommunity(0, -1)">Up</button>
    <button onclick="nudgeSelectedCommunity(0, 1)">Down</button>
    <button onclick="spreadSelectedCommunity()">Spread</button>
    <button onclick="tightenSelectedCommunity()">Tighten</button>
    <label>Step <input id="nudgeStep" type="range" min="20" max="260" step="20" value="100"></label>
    <label>Spread <input id="spreadStep" type="range" min="1.05" max="2.50" step="0.05" value="1.35"></label>
    <label>Node <input id="nodeScale" type="range" min="0.5" max="2.6" step="0.1" value="1.0" oninput="updateSizes()"></label>
    <label>Label <input id="labelScale" type="range" min="0.5" max="3.2" step="0.1" value="1.0" oninput="updateSizes()"></label>
  </div>
  <div id="network-chart"></div>

  <script>
    const DATA_FILE = "__DATA_FILE__";
    const SERIES_ID = "epb-network-series";
    const chart = echarts.init(document.getElementById("network-chart"), null, { renderer: "canvas" });
    let networkData = null;
    let baseNodes = [];
    let currentNodes = [];
    let showLabels = true;
    let manualLayout = false;

    chart.showLoading({
      text: "Loading network data...",
      color: "#2563eb",
      textColor: "#1f2937",
      maskColor: "rgba(255,255,255,0.85)"
    });

    fetch(DATA_FILE)
      .then(response => {
        if (!response.ok) throw new Error("Cannot load " + DATA_FILE);
        return response.json();
      })
      .then(data => {
        networkData = data;
        assignCommunityColors(networkData);
        baseNodes = deepCopy(networkData.nodes);
        currentNodes = deepCopy(networkData.nodes);
        populateCommunitySelect();
        chart.hideLoading();
        setForceOption();
      })
      .catch(error => {
        chart.hideLoading();
        const message = "Network data was not loaded. Open this HTML through a local server such as VS Code Live Server. " + error.message;
        document.getElementById("network-chart").innerHTML = "<div style='padding:24px;font:16px Arial;color:#b91c1c;'>" + message + "</div>";
      });

    window.addEventListener("resize", () => chart.resize());
    chart.on("mouseup", () => {
      if (manualLayout) {
        currentNodes = markNodesFixed(captureCurrentNodes(true));
      }
    });

    function deepCopy(value) {
      return JSON.parse(JSON.stringify(value));
    }

    const ECHARTS_STYLE_PALETTE = [
      "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272",
      "#fc8452", "#9a60b4", "#ea7ccc", "#2f4554", "#61a0a8", "#d48265",
      "#749f83", "#ca8622", "#bda29a", "#6e7074", "#546570", "#c4ccd3",
      "#6f8edc", "#a7d88a", "#ffd976", "#f27c7c", "#8ad0ea", "#55b487",
      "#fd9b70", "#b07ac6", "#ef91d2", "#405f70", "#79b3ba", "#dd987d",
      "#8db796", "#d59a45", "#cbb8af", "#85878b", "#6d7f8d", "#d3d9df"
    ];

    function assignCommunityColors(data) {
      const count = Math.max(data.categories.length, 1);
      data.categories.forEach((category, index) => {
        category.itemStyle = Object.assign({}, category.itemStyle || {}, {
          color: communityColor(index, count)
        });
      });
      data.nodes.forEach(node => {
        const category = data.categories[node.category] || {};
        const color = category.itemStyle ? category.itemStyle.color : communityColor(node.category || 0, count);
        node.itemStyle = Object.assign({}, node.itemStyle || {}, {
          color: color,
          borderColor: "#ffffff",
          borderWidth: 1
        });
      });
    }

    function communityColor(index, count) {
      if (index < ECHARTS_STYLE_PALETTE.length) {
        return ECHARTS_STYLE_PALETTE[index];
      }
      return ECHARTS_STYLE_PALETTE[index % ECHARTS_STYLE_PALETTE.length];
    }

    function forceConfig() {
      return {
        repulsion: 2000,
        edgeLength: [100, 400],
        gravity: 0.02,
        friction: 0.68,
        layoutAnimation: true
      };
    }

    function baseSeries(extra) {
      return Object.assign({
        id: SERIES_ID,
        name: "EPB Network",
        type: "graph",
        data: currentNodes,
        links: networkData.links,
        categories: networkData.categories,
        roam: true,
        draggable: true,
        focusNodeAdjacency: true,
        itemStyle: {
          borderColor: "#ffffff",
          borderWidth: 1
        },
        lineStyle: {
          color: "source",
          curveness: 0.08,
          opacity: 0.35
        },
        label: {
          show: showLabels,
          position: "right",
          formatter: "{b}",
          color: "#1f2937",
          fontWeight: "bold",
          textBorderColor: "#ffffff",
          textBorderWidth: 3
        },
        labelLayout: {
          hideOverlap: false
        },
        scaleLimit: {
          min: 0.02,
          max: 20
        },
        emphasis: {
          focus: "adjacency",
          lineStyle: {
            opacity: 0.9,
            width: 4
          }
        }
      }, extra || {});
    }

    function updateSeries(partial, options) {
      chart.setOption({
        series: [Object.assign({ id: SERIES_ID }, partial)]
      }, Object.assign({ notMerge: false, lazyUpdate: true }, options || {}));
    }

    function setForceOption() {
      chart.setOption({
        backgroundColor: "#ffffff",
        tooltip: {
          trigger: "item",
          formatter: tooltipText,
          backgroundColor: "rgba(255,255,255,0.96)",
          borderColor: "#94a3b8",
          borderWidth: 1,
          textStyle: { color: "#1f2937", fontSize: 12 }
        },
        legend: [{ data: networkData.categories.map(item => item.name), show: false }],
        series: [baseSeries({ layout: "force", force: forceConfig() })]
      }, true);
    }

    function setManualOption() {
      updateManualData();
    }

    function tooltipText(params) {
      if (params.dataType === "node") {
        const category = networkData.categories[params.data.category] || { name: "Unknown" };
        return [
          "<strong>" + params.data.full_name + "</strong>",
          "Papers: " + params.data.papers,
          "Weighted Degree: " + params.data.weighted_degree,
          "Community: " + category.name
        ].join("<br>");
      }
      if (params.dataType === "edge") {
        return [
          "<strong>Collaboration</strong>",
          "Weight: " + params.data.value,
          "Between: " + params.data.source + " - " + params.data.target
        ].join("<br>");
      }
      return "";
    }

    function populateCommunitySelect() {
      const select = document.getElementById("communitySelect");
      select.innerHTML = "";
      networkData.categories.forEach((category, index) => {
        const count = networkData.nodes.filter(node => node.category === index).length;
        if (count === 0) return;
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = category.name + " (" + count + ")";
        select.appendChild(option);
      });
    }

    function captureCurrentNodes(fixed) {
      if (!networkData) return [];
      const series = chart.getModel().getSeriesByIndex(0);
      if (!series) return currentNodes;
      const data = series.getData();
      return currentNodes.map((node, index) => {
        const next = Object.assign({}, node);
        const layout = data.getItemLayout(index);
        if (layout && Number.isFinite(layout[0]) && Number.isFinite(layout[1])) {
          next.x = layout[0];
          next.y = layout[1];
        }
        next.fixed = fixed;
        return next;
      });
    }

    function markNodesFixed(nodes) {
      return nodes.map(node => Object.assign({}, node, { fixed: true }));
    }

    function layoutStats(nodes) {
      const xs = nodes.map(node => node.x).filter(Number.isFinite);
      const ys = nodes.map(node => node.y).filter(Number.isFinite);
      return {
        finiteX: xs.length,
        finiteY: ys.length,
        xSpread: xs.length ? Math.max.apply(null, xs) - Math.min.apply(null, xs) : 0,
        ySpread: ys.length ? Math.max.apply(null, ys) - Math.min.apply(null, ys) : 0
      };
    }

    function layoutIsReady(nodes) {
      const stats = layoutStats(nodes);
      const minFinite = Math.floor(nodes.length * 0.95);
      const minSpread = Math.min(chart.getWidth(), chart.getHeight()) * 0.45;
      return stats.finiteX >= minFinite
        && stats.finiteY >= minFinite
        && stats.xSpread > minSpread
        && stats.ySpread > minSpread;
    }

    function updateManualData() {
      if (!layoutIsReady(currentNodes)) return;
      currentNodes = markNodesFixed(currentNodes);
      updateSeries({
        layout: "none",
        data: currentNodes,
        animation: false,
        force: { layoutAnimation: false }
      }, { lazyUpdate: false });
    }

    function toggleManualLayout() {
      if (!networkData) return;
      const button = document.getElementById("lockBtn");
      if (!manualLayout) {
        enterManualLayout();
      } else {
        currentNodes = captureCurrentNodes(false).map(node => {
          const next = Object.assign({}, node);
          delete next.x;
          delete next.y;
          next.fixed = false;
          return next;
        });
        manualLayout = false;
        button.textContent = "Lock Layout";
        button.classList.remove("active");
        updateSeries({
          layout: "force",
          data: currentNodes,
          force: forceConfig()
        }, { lazyUpdate: false });
      }
    }

    function enterManualLayout(onReady, attempt) {
      const button = document.getElementById("lockBtn");
      const tryIndex = attempt || 0;
      if (manualLayout) {
        if (onReady) onReady();
        return true;
      }

      const captured = captureCurrentNodes(true);
      if (layoutIsReady(captured)) {
        currentNodes = markNodesFixed(captured);
        manualLayout = true;
        button.textContent = "Resume Force";
        button.classList.add("active");
        updateManualData();
        if (onReady) setTimeout(onReady, 0);
        return true;
      }

      if (tryIndex < 20) {
        button.textContent = "Waiting Layout";
        setTimeout(() => enterManualLayout(onReady, tryIndex + 1), 200);
        return false;
      }

      button.textContent = "Lock Layout";
      alert("The force layout is not ready to lock yet. Wait a few seconds, then click Lock Layout again.");
      return false;
    }

    function selectedCommunityId() {
      return Number(document.getElementById("communitySelect").value);
    }

    function updateSelectionInfo() {
      return selectedCommunityId();
    }

    function moveSelectedCommunity(corner) {
      if (!networkData) return;
      enterManualLayout(() => {
        const category = selectedCommunityId();
        const nodes = currentNodes.filter(node => node.category === category);
        if (nodes.length === 0) return;
        const box = boundingBox(nodes);
        const targets = {
          "top-left": [chart.getWidth() * 0.16, chart.getHeight() * 0.18],
          "top-right": [chart.getWidth() * 0.84, chart.getHeight() * 0.18],
          "bottom-left": [chart.getWidth() * 0.16, chart.getHeight() * 0.82],
          "bottom-right": [chart.getWidth() * 0.84, chart.getHeight() * 0.82]
        };
        const target = targets[corner];
        translateCommunity(category, target[0] - box.cx, target[1] - box.cy);
      });
    }

    function nudgeSelectedCommunity(dx, dy) {
      if (!networkData) return;
      enterManualLayout(() => {
        const step = Number(document.getElementById("nudgeStep").value);
        translateCommunity(selectedCommunityId(), dx * step, dy * step);
      });
    }

    function spreadSelectedCommunity() {
      if (!networkData) return;
      const factor = Number(document.getElementById("spreadStep").value);
      scaleCommunity(selectedCommunityId(), factor);
    }

    function tightenSelectedCommunity() {
      if (!networkData) return;
      const factor = Number(document.getElementById("spreadStep").value);
      scaleCommunity(selectedCommunityId(), 1 / factor);
    }

    function scaleCommunity(category, factor) {
      enterManualLayout(() => {
        const nodes = currentNodes.filter(node => node.category === category);
        if (nodes.length === 0) return;
        const box = boundingBox(nodes);
        currentNodes = currentNodes.map(node => {
          if (node.category !== category) return node;
          const x = Number.isFinite(node.x) ? node.x : box.cx;
          const y = Number.isFinite(node.y) ? node.y : box.cy;
          return Object.assign({}, node, {
            x: box.cx + (x - box.cx) * factor,
            y: box.cy + (y - box.cy) * factor,
            fixed: true
          });
        });
        relaxCommunityCollisions(category, 8);
        updateManualData();
      });
    }

    function relaxCommunityCollisions(category, iterations) {
      for (let iter = 0; iter < iterations; iter += 1) {
        const indexes = [];
        currentNodes.forEach((node, index) => {
          if (node.category === category) indexes.push(index);
        });
        for (let a = 0; a < indexes.length; a += 1) {
          for (let b = a + 1; b < indexes.length; b += 1) {
            const i = indexes[a];
            const j = indexes[b];
            const nodeA = currentNodes[i];
            const nodeB = currentNodes[j];
            const radiusA = (nodeA.symbolSize || 8) * 0.85 + labelRadius(nodeA);
            const radiusB = (nodeB.symbolSize || 8) * 0.85 + labelRadius(nodeB);
            const minDistance = radiusA + radiusB + 10;
            let dx = nodeB.x - nodeA.x;
            let dy = nodeB.y - nodeA.y;
            let distance = Math.sqrt(dx * dx + dy * dy);
            if (!Number.isFinite(distance) || distance < 0.001) {
              const angle = ((i + j + iter) % 360) * Math.PI / 180;
              dx = Math.cos(angle);
              dy = Math.sin(angle);
              distance = 1;
            }
            if (distance >= minDistance) continue;
            const push = (minDistance - distance) * 0.5;
            const ux = dx / distance;
            const uy = dy / distance;
            currentNodes[i] = Object.assign({}, nodeA, {
              x: nodeA.x - ux * push,
              y: nodeA.y - uy * push,
              fixed: true
            });
            currentNodes[j] = Object.assign({}, nodeB, {
              x: nodeB.x + ux * push,
              y: nodeB.y + uy * push,
              fixed: true
            });
          }
        }
      }
    }

    function labelRadius(node) {
      const label = node.label || {};
      const fontSize = label.fontSize || 12;
      const text = node.name || "";
      if (!showLabels || label.show === false || text.length === 0) return 0;
      return Math.min(90, text.length * fontSize * 0.32);
    }

    function translateCommunity(category, dx, dy) {
      currentNodes = currentNodes.map(node => {
        if (node.category !== category) return node;
        return Object.assign({}, node, {
          x: (Number.isFinite(node.x) ? node.x : 0) + dx,
          y: (Number.isFinite(node.y) ? node.y : 0) + dy,
          fixed: true
        });
      });
      updateManualData();
    }

    function boundingBox(nodes) {
      const xs = nodes.map(node => Number.isFinite(node.x) ? node.x : 0);
      const ys = nodes.map(node => Number.isFinite(node.y) ? node.y : 0);
      const minX = Math.min.apply(null, xs);
      const maxX = Math.max.apply(null, xs);
      const minY = Math.min.apply(null, ys);
      const maxY = Math.max.apply(null, ys);
      return {
        minX,
        maxX,
        minY,
        maxY,
        cx: (minX + maxX) / 2,
        cy: (minY + maxY) / 2
      };
    }

    function resetView() {
      chart.dispatchAction({ type: "restore" });
    }

    function toggleLabels() {
      showLabels = !showLabels;
      const patch = { label: { show: showLabels } };
      if (manualLayout) patch.layout = "none";
      updateSeries(patch, { lazyUpdate: false });
    }

    function updateSizes() {
      if (!networkData) return;
      enterManualLayout(() => {
        const nodeScale = Number(document.getElementById("nodeScale").value);
        const labelScale = Number(document.getElementById("labelScale").value);
        const baseById = new Map(baseNodes.map(node => [node.id, node]));
        currentNodes = currentNodes.map(node => {
          const base = baseById.get(node.id) || node;
          const next = Object.assign({}, node);
          next.symbolSize = Math.max(2, Math.round((base.symbolSize || 8) * nodeScale));
          if (base.label) {
            next.label = Object.assign({}, base.label, {
              show: showLabels,
              fontSize: Math.max(6, Math.round((base.label.fontSize || 12) * labelScale))
            });
          }
          return next;
        });
        updateManualData();
      });
    }

    function saveAsImage() {
      const url = chart.getDataURL({
        type: "png",
        pixelRatio: 3,
        backgroundColor: "#ffffff"
      });
      const link = document.createElement("a");
      link.download = "epb-network.png";
      link.href = url;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  </script>
</body>
</html>
"""
    html_path.write_text(html_text.replace("__DATA_FILE__", data_file), encoding="utf-8")


def write_community_pack_network_html(html_path: Path, data_file: str):
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EPB Community-Packed Network</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #ffffff;
      color: #1f2937;
      font-family: Arial, Helvetica, sans-serif;
    }
    .toolbar {
      height: 52px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      box-sizing: border-box;
      border-bottom: 1px solid #e5e7eb;
      background: #ffffff;
      white-space: nowrap;
      overflow-x: auto;
    }
    button, select, input {
      font: inherit;
    }
    button {
      height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #f8fafc;
      color: #111827;
      padding: 0 10px;
      cursor: pointer;
    }
    button:hover {
      background: #eef2f7;
    }
    select {
      height: 34px;
      min-width: 160px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: #111827;
      padding: 0 8px;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #475569;
      font-size: 13px;
    }
    input[type="range"] {
      width: 96px;
    }
    #network-chart {
      width: 100vw;
      height: calc(100vh - 52px);
      background: #ffffff;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button onclick="repackCommunities()">Auto Pack</button>
    <button onclick="spreadAllCommunities()">Spread All</button>
    <button onclick="resetView()">Reset View</button>
    <button onclick="toggleLabels()">Toggle Labels</button>
    <button onclick="saveAsImage()">Save PNG</button>
    <select id="communitySelect"></select>
    <button onclick="moveSelectedCommunity('top-left')">Top Left</button>
    <button onclick="moveSelectedCommunity('top-right')">Top Right</button>
    <button onclick="moveSelectedCommunity('bottom-left')">Bottom Left</button>
    <button onclick="moveSelectedCommunity('bottom-right')">Bottom Right</button>
    <button onclick="nudgeSelectedCommunity(-1, 0)">Left</button>
    <button onclick="nudgeSelectedCommunity(1, 0)">Right</button>
    <button onclick="nudgeSelectedCommunity(0, -1)">Up</button>
    <button onclick="nudgeSelectedCommunity(0, 1)">Down</button>
    <button onclick="spreadSelectedCommunity()">Spread</button>
    <button onclick="tightenSelectedCommunity()">Tighten</button>
    <label>Step <input id="nudgeStep" type="range" min="20" max="260" step="20" value="100"></label>
    <label>Spread <input id="spreadStep" type="range" min="1.05" max="2.50" step="0.05" value="1.30"></label>
    <label>Node <input id="nodeScale" type="range" min="0.5" max="2.6" step="0.1" value="1.0" oninput="updateSizes()"></label>
    <label>Label <input id="labelScale" type="range" min="0.5" max="3.2" step="0.1" value="1.0" oninput="updateSizes()"></label>
  </div>
  <div id="network-chart"></div>

  <script>
    const DATA_FILE = "__DATA_FILE__";
    const SERIES_ID = "epb-community-pack-series";
    const ECHARTS_STYLE_PALETTE = [
      "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272",
      "#fc8452", "#9a60b4", "#ea7ccc", "#2f4554", "#61a0a8", "#d48265",
      "#749f83", "#ca8622", "#bda29a", "#6e7074", "#546570", "#c4ccd3",
      "#6f8edc", "#a7d88a", "#ffd976", "#f27c7c", "#8ad0ea", "#55b487",
      "#fd9b70", "#b07ac6", "#ef91d2", "#405f70", "#79b3ba", "#dd987d",
      "#8db796", "#d59a45", "#cbb8af", "#85878b", "#6d7f8d", "#d3d9df"
    ];

    const chart = echarts.init(document.getElementById("network-chart"), null, { renderer: "canvas" });
    let networkData = null;
    let baseNodes = [];
    let currentNodes = [];
    let communityRects = new Map();
    let showLabels = true;

    chart.showLoading({
      text: "Loading network data...",
      color: "#5470c6",
      textColor: "#1f2937",
      maskColor: "rgba(255,255,255,0.85)"
    });

    fetch(DATA_FILE)
      .then(response => {
        if (!response.ok) throw new Error("Cannot load " + DATA_FILE);
        return response.json();
      })
      .then(data => {
        networkData = data;
        assignCommunityColors(networkData);
        baseNodes = deepCopy(networkData.nodes).map(node => {
          const baseFont = node.label && node.label.fontSize ? node.label.fontSize : 12;
          node._baseSymbolSize = node.symbolSize || 8;
          node._baseFontSize = baseFont;
          return node;
        });
        populateCommunitySelect();
        chart.hideLoading();
        repackCommunities();
      })
      .catch(error => {
        chart.hideLoading();
        const message = "Network data was not loaded. Open this HTML through a local server such as VS Code Live Server. " + error.message;
        document.getElementById("network-chart").innerHTML = "<div style='padding:24px;font:16px Arial;color:#b91c1c;'>" + message + "</div>";
      });

    window.addEventListener("resize", () => {
      chart.resize();
      if (networkData) repackCommunities();
    });
    chart.on("mouseup", () => {
      currentNodes = captureCurrentNodes();
    });

    function deepCopy(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function assignCommunityColors(data) {
      data.categories.forEach((category, index) => {
        category.itemStyle = Object.assign({}, category.itemStyle || {}, {
          color: ECHARTS_STYLE_PALETTE[index % ECHARTS_STYLE_PALETTE.length]
        });
      });
      data.nodes.forEach(node => {
        const category = data.categories[node.category] || {};
        const color = category.itemStyle ? category.itemStyle.color : ECHARTS_STYLE_PALETTE[0];
        node.itemStyle = Object.assign({}, node.itemStyle || {}, {
          color: color,
          borderColor: "#ffffff",
          borderWidth: 1
        });
      });
    }

    function populateCommunitySelect() {
      const select = document.getElementById("communitySelect");
      select.innerHTML = "";
      networkData.categories.forEach((category, index) => {
        const count = networkData.nodes.filter(node => node.category === index).length;
        if (count === 0) return;
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = category.name + " (" + count + ")";
        select.appendChild(option);
      });
    }

    function selectedCommunityId() {
      return Number(document.getElementById("communitySelect").value);
    }

    function repackCommunities() {
      if (!networkData) return;
      const nodesByCommunity = groupNodesByCommunity(baseNodes);
      const communities = Array.from(nodesByCommunity.entries()).map(([category, nodes]) => ({
        category,
        nodes,
        weight: Math.max(nodes.length, 1)
      })).sort((a, b) => b.weight - a.weight || a.category - b.category);

      communityRects = new Map();
      const root = {
        x: 58,
        y: 58,
        w: Math.max(600, chart.getWidth() - 116),
        h: Math.max(420, chart.getHeight() - 116)
      };
      splitCommunityRects(communities, root, 0);

      const nextNodes = deepCopy(baseNodes);
      const byId = new Map(nextNodes.map(node => [node.id, node]));
      communities.forEach(community => {
        const rect = communityRects.get(community.category);
        const nodes = community.nodes.map(node => byId.get(node.id)).filter(Boolean);
        layoutNodesInsideCommunity(community.category, nodes, rect);
      });
      currentNodes = nextNodes.map(node => Object.assign({}, node, { fixed: true }));
      applySizeScales();
      renderNetwork(true);
    }

    function groupNodesByCommunity(nodes) {
      const grouped = new Map();
      nodes.forEach(node => {
        if (!grouped.has(node.category)) grouped.set(node.category, []);
        grouped.get(node.category).push(node);
      });
      return grouped;
    }

    function splitCommunityRects(items, rect, depth) {
      if (items.length === 0) return;
      if (items.length === 1) {
        const pad = Math.max(16, Math.min(rect.w, rect.h) * 0.045);
        communityRects.set(items[0].category, {
          x: rect.x + pad,
          y: rect.y + pad,
          w: Math.max(80, rect.w - 2 * pad),
          h: Math.max(80, rect.h - 2 * pad)
        });
        return;
      }

      const total = items.reduce((sum, item) => sum + item.weight, 0);
      let bestIndex = 1;
      let bestDiff = Infinity;
      let acc = 0;
      for (let i = 1; i < items.length; i += 1) {
        acc += items[i - 1].weight;
        const diff = Math.abs(total / 2 - acc);
        if (diff < bestDiff) {
          bestDiff = diff;
          bestIndex = i;
        }
      }

      const first = items.slice(0, bestIndex);
      const second = items.slice(bestIndex);
      const firstWeight = first.reduce((sum, item) => sum + item.weight, 0);
      const ratio = Math.max(0.18, Math.min(0.82, firstWeight / total));
      const gap = 18;

      if (rect.w >= rect.h) {
        const firstW = (rect.w - gap) * ratio;
        splitCommunityRects(first, { x: rect.x, y: rect.y, w: firstW, h: rect.h }, depth + 1);
        splitCommunityRects(second, { x: rect.x + firstW + gap, y: rect.y, w: rect.w - firstW - gap, h: rect.h }, depth + 1);
      } else {
        const firstH = (rect.h - gap) * ratio;
        splitCommunityRects(first, { x: rect.x, y: rect.y, w: rect.w, h: firstH }, depth + 1);
        splitCommunityRects(second, { x: rect.x, y: rect.y + firstH + gap, w: rect.w, h: rect.h - firstH - gap }, depth + 1);
      }
    }

    function layoutNodesInsideCommunity(category, nodes, rect) {
      if (!rect || nodes.length === 0) return;
      const cx = rect.x + rect.w / 2;
      const cy = rect.y + rect.h / 2;
      const sorted = nodes.slice().sort((a, b) =>
        (b.weighted_degree || 0) - (a.weighted_degree || 0) || String(a.name).localeCompare(String(b.name))
      );
      const maxRx = Math.max(26, rect.w * 0.43);
      const maxRy = Math.max(26, rect.h * 0.43);
      sorted.forEach((node, index) => {
        if (sorted.length === 1) {
          node.x = cx;
          node.y = cy;
        } else {
          const t = index / Math.max(sorted.length - 1, 1);
          const radius = Math.sqrt(t);
          const angle = index * 2.399963229728653;
          node.x = cx + Math.cos(angle) * maxRx * radius;
          node.y = cy + Math.sin(angle) * maxRy * radius;
        }
        node.fixed = true;
      });

      const localLinks = communityLinks(category, sorted);
      relaxLocalLayout(sorted, rect, localLinks, 180);
    }

    function communityLinks(category, nodes) {
      const ids = new Set(nodes.map(node => node.id));
      const indexById = new Map(nodes.map((node, index) => [node.id, index]));
      return networkData.links
        .filter(link => ids.has(String(link.source)) && ids.has(String(link.target)))
        .map(link => ({
          source: indexById.get(String(link.source)),
          target: indexById.get(String(link.target)),
          weight: Number(link.value || link.weight || 1)
        }))
        .filter(link => Number.isFinite(link.source) && Number.isFinite(link.target));
    }

    function relaxLocalLayout(nodes, rect, links, iterations) {
      for (let iter = 0; iter < iterations; iter += 1) {
        for (let a = 0; a < nodes.length; a += 1) {
          for (let b = a + 1; b < nodes.length; b += 1) {
            pushPairApart(nodes[a], nodes[b], iter, 0.58);
          }
        }

        links.forEach(link => {
          const a = nodes[link.source];
          const b = nodes[link.target];
          if (!a || !b) return;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
          const ideal = 62 + Math.min(42, (labelRadius(a) + labelRadius(b)) * 0.18);
          const pull = Math.max(-7, Math.min(7, (distance - ideal) * 0.012 * Math.min(2.5, link.weight || 1)));
          const ux = dx / distance;
          const uy = dy / distance;
          a.x += ux * pull;
          a.y += uy * pull;
          b.x -= ux * pull;
          b.y -= uy * pull;
        });

        const cx = rect.x + rect.w / 2;
        const cy = rect.y + rect.h / 2;
        nodes.forEach(node => {
          node.x += (cx - node.x) * 0.004;
          node.y += (cy - node.y) * 0.004;
          clampNodeToRect(node, rect);
        });
      }
    }

    function pushPairApart(a, b, iter, strength) {
      const minDistance = nodeRadius(a) + nodeRadius(b) + 8;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      let distance = Math.sqrt(dx * dx + dy * dy);
      if (!Number.isFinite(distance) || distance < 0.001) {
        const angle = ((Number(a.id) + Number(b.id) + iter) % 360) * Math.PI / 180;
        dx = Math.cos(angle);
        dy = Math.sin(angle);
        distance = 1;
      }
      if (distance >= minDistance) return;
      const push = (minDistance - distance) * 0.5 * strength;
      const ux = dx / distance;
      const uy = dy / distance;
      a.x -= ux * push;
      a.y -= uy * push;
      b.x += ux * push;
      b.y += uy * push;
    }

    function clampNodeToRect(node, rect) {
      const margin = Math.min(90, Math.max(18, nodeRadius(node) * 0.55));
      node.x = Math.max(rect.x + margin, Math.min(rect.x + rect.w - margin, node.x));
      node.y = Math.max(rect.y + margin, Math.min(rect.y + rect.h - margin, node.y));
    }

    function nodeRadius(node) {
      return (node.symbolSize || node._baseSymbolSize || 8) * 0.62 + labelRadius(node);
    }

    function labelRadius(node) {
      const label = node.label || {};
      const fontSize = label.fontSize || node._baseFontSize || 12;
      const text = node.name || "";
      if (!showLabels || label.show === false || text.length === 0) return 0;
      return Math.min(110, text.length * fontSize * 0.31);
    }

    function applySizeScales() {
      const nodeScale = Number(document.getElementById("nodeScale").value);
      const labelScale = Number(document.getElementById("labelScale").value);
      currentNodes = currentNodes.map(node => {
        const next = Object.assign({}, node);
        next.symbolSize = Math.max(2, Math.round((node._baseSymbolSize || node.symbolSize || 8) * nodeScale));
        const baseLabel = node.label || {};
        next.label = Object.assign({}, baseLabel, {
          show: showLabels,
          fontSize: Math.max(6, Math.round((node._baseFontSize || baseLabel.fontSize || 12) * labelScale))
        });
        return next;
      });
    }

    function renderNetwork(reset) {
      chart.setOption({
        backgroundColor: "#ffffff",
        tooltip: {
          trigger: "item",
          formatter: tooltipText,
          backgroundColor: "rgba(255,255,255,0.96)",
          borderColor: "#94a3b8",
          borderWidth: 1,
          textStyle: { color: "#1f2937", fontSize: 12 }
        },
        legend: [{ data: networkData.categories.map(item => item.name), show: false }],
        series: [{
          id: SERIES_ID,
          name: "EPB Community Pack",
          type: "graph",
          layout: "none",
          data: currentNodes,
          links: networkData.links,
          categories: networkData.categories,
          left: 0,
          top: 0,
          right: 0,
          bottom: 0,
          roam: true,
          draggable: true,
          focusNodeAdjacency: true,
          animation: false,
          itemStyle: {
            borderColor: "#ffffff",
            borderWidth: 1
          },
          lineStyle: {
            color: "source",
            curveness: 0.04,
            opacity: 0.28
          },
          label: {
            show: showLabels,
            position: "right",
            formatter: "{b}",
            color: "#1f2937",
            fontWeight: "bold",
            textBorderColor: "#ffffff",
            textBorderWidth: 3
          },
          labelLayout: {
            hideOverlap: false
          },
          scaleLimit: {
            min: 0.02,
            max: 20
          },
          emphasis: {
            focus: "adjacency",
            lineStyle: {
              opacity: 0.85,
              width: 4
            }
          }
        }]
      }, Boolean(reset));
    }

    function tooltipText(params) {
      if (params.dataType === "node") {
        const category = networkData.categories[params.data.category] || { name: "Unknown" };
        return [
          "<strong>" + params.data.full_name + "</strong>",
          "Papers: " + params.data.papers,
          "Weighted Degree: " + params.data.weighted_degree,
          "Community: " + category.name
        ].join("<br>");
      }
      if (params.dataType === "edge") {
        return [
          "<strong>Collaboration</strong>",
          "Weight: " + params.data.value,
          "Between: " + params.data.source + " - " + params.data.target
        ].join("<br>");
      }
      return "";
    }

    function captureCurrentNodes() {
      const series = chart.getModel().getSeriesByIndex(0);
      if (!series) return currentNodes;
      const data = series.getData();
      return currentNodes.map((node, index) => {
        const next = Object.assign({}, node);
        const layout = data.getItemLayout(index);
        if (layout && Number.isFinite(layout[0]) && Number.isFinite(layout[1])) {
          next.x = layout[0];
          next.y = layout[1];
        }
        next.fixed = true;
        return next;
      });
    }

    function selectedCommunityNodes() {
      const category = selectedCommunityId();
      return currentNodes.filter(node => node.category === category);
    }

    function moveSelectedCommunity(corner) {
      const nodes = selectedCommunityNodes();
      if (nodes.length === 0) return;
      const box = boundingBox(nodes);
      const targets = {
        "top-left": [chart.getWidth() * 0.16, chart.getHeight() * 0.18],
        "top-right": [chart.getWidth() * 0.84, chart.getHeight() * 0.18],
        "bottom-left": [chart.getWidth() * 0.16, chart.getHeight() * 0.82],
        "bottom-right": [chart.getWidth() * 0.84, chart.getHeight() * 0.82]
      };
      const target = targets[corner];
      translateCommunity(selectedCommunityId(), target[0] - box.cx, target[1] - box.cy);
    }

    function nudgeSelectedCommunity(dx, dy) {
      const step = Number(document.getElementById("nudgeStep").value);
      translateCommunity(selectedCommunityId(), dx * step, dy * step);
    }

    function translateCommunity(category, dx, dy) {
      currentNodes = currentNodes.map(node => {
        if (node.category !== category) return node;
        return Object.assign({}, node, {
          x: node.x + dx,
          y: node.y + dy,
          fixed: true
        });
      });
      renderNetwork(false);
    }

    function spreadSelectedCommunity() {
      scaleCommunity(selectedCommunityId(), Number(document.getElementById("spreadStep").value), 14);
    }

    function tightenSelectedCommunity() {
      scaleCommunity(selectedCommunityId(), 1 / Number(document.getElementById("spreadStep").value), 6);
    }

    function spreadAllCommunities() {
      const categories = Array.from(new Set(currentNodes.map(node => node.category)));
      categories.forEach(category => scaleCommunity(category, 1.12, 5, false));
      renderNetwork(false);
    }

    function scaleCommunity(category, factor, collisionIterations, rerender) {
      const nodes = currentNodes.filter(node => node.category === category);
      if (nodes.length === 0) return;
      const box = boundingBox(nodes);
      currentNodes = currentNodes.map(node => {
        if (node.category !== category) return node;
        return Object.assign({}, node, {
          x: box.cx + (node.x - box.cx) * factor,
          y: box.cy + (node.y - box.cy) * factor,
          fixed: true
        });
      });
      const active = currentNodes.filter(node => node.category === category);
      relaxLocalLayout(active, boundingBoxRect(active), [], collisionIterations || 8);
      const activeById = new Map(active.map(node => [node.id, node]));
      currentNodes = currentNodes.map(node => activeById.get(node.id) || node);
      if (rerender !== false) renderNetwork(false);
    }

    function boundingBox(nodes) {
      const xs = nodes.map(node => node.x).filter(Number.isFinite);
      const ys = nodes.map(node => node.y).filter(Number.isFinite);
      const minX = Math.min.apply(null, xs);
      const maxX = Math.max.apply(null, xs);
      const minY = Math.min.apply(null, ys);
      const maxY = Math.max.apply(null, ys);
      return {
        minX,
        maxX,
        minY,
        maxY,
        cx: (minX + maxX) / 2,
        cy: (minY + maxY) / 2
      };
    }

    function boundingBoxRect(nodes) {
      const box = boundingBox(nodes);
      const pad = 80;
      return {
        x: box.minX - pad,
        y: box.minY - pad,
        w: Math.max(160, box.maxX - box.minX + 2 * pad),
        h: Math.max(160, box.maxY - box.minY + 2 * pad)
      };
    }

    function resetView() {
      chart.dispatchAction({ type: "restore" });
    }

    function toggleLabels() {
      showLabels = !showLabels;
      applySizeScales();
      renderNetwork(false);
    }

    function updateSizes() {
      currentNodes = captureCurrentNodes();
      applySizeScales();
      renderNetwork(false);
    }

    function saveAsImage() {
      const url = chart.getDataURL({
        type: "png",
        pixelRatio: 3,
        backgroundColor: "#ffffff"
      });
      const link = document.createElement("a");
      link.download = "epb-community-pack-network.png";
      link.href = url;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  </script>
</body>
</html>
"""
    html_path.write_text(html_text.replace("__DATA_FILE__", data_file), encoding="utf-8")


def write_hybrid_force_pack_network_html(html_path: Path, data_file: str):
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EPB Hybrid Force Network</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #ffffff;
      color: #1f2937;
      font-family: Arial, Helvetica, sans-serif;
    }
    .toolbar {
      height: 52px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      box-sizing: border-box;
      border-bottom: 1px solid #e5e7eb;
      background: #ffffff;
      white-space: nowrap;
      overflow-x: auto;
    }
    button, select, input {
      font: inherit;
    }
    button {
      height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #f8fafc;
      color: #111827;
      padding: 0 10px;
      cursor: pointer;
    }
    button:hover {
      background: #eef2f7;
    }
    select {
      height: 34px;
      min-width: 160px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: #111827;
      padding: 0 8px;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #475569;
      font-size: 13px;
    }
    input[type="range"] {
      width: 88px;
    }
    #network-chart {
      width: 100vw;
      height: calc(100vh - 52px);
      background: #ffffff;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button onclick="runHybridLayout(true)">Hybrid Pack</button>
    <button onclick="runHybridLayout(false)">Refine Force</button>
    <button onclick="stopSimulation()">Stop</button>
    <button onclick="resetView()">Reset View</button>
    <button onclick="toggleLabels()">Toggle Labels</button>
    <button onclick="saveAsImage()">Save PNG</button>
    <select id="communitySelect"></select>
    <button onclick="moveSelectedCommunity('top-left')">Top Left</button>
    <button onclick="moveSelectedCommunity('top-right')">Top Right</button>
    <button onclick="moveSelectedCommunity('bottom-left')">Bottom Left</button>
    <button onclick="moveSelectedCommunity('bottom-right')">Bottom Right</button>
    <button onclick="nudgeSelectedCommunity(-1, 0)">Left</button>
    <button onclick="nudgeSelectedCommunity(1, 0)">Right</button>
    <button onclick="nudgeSelectedCommunity(0, -1)">Up</button>
    <button onclick="nudgeSelectedCommunity(0, 1)">Down</button>
    <button onclick="spreadSelectedCommunity()">Spread</button>
    <button onclick="tightenSelectedCommunity()">Tighten</button>
    <label>Fill <input id="fillScale" type="range" min="0.65" max="1.08" step="0.03" value="0.94"></label>
    <label>Pull <input id="anchorPull" type="range" min="0.015" max="0.11" step="0.005" value="0.055"></label>
    <label>Step <input id="nudgeStep" type="range" min="20" max="260" step="20" value="100"></label>
    <label>Spread <input id="spreadStep" type="range" min="1.05" max="2.50" step="0.05" value="1.25"></label>
    <label>Node <input id="nodeScale" type="range" min="0.5" max="2.6" step="0.1" value="1.0" oninput="updateSizes()"></label>
    <label>Label <input id="labelScale" type="range" min="0.5" max="3.2" step="0.1" value="1.0" oninput="updateSizes()"></label>
  </div>
  <div id="network-chart"></div>

  <script>
    const DATA_FILE = "__DATA_FILE__";
    const SERIES_ID = "epb-hybrid-force-series";
    const ECHARTS_STYLE_PALETTE = [
      "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272",
      "#fc8452", "#9a60b4", "#ea7ccc", "#2f4554", "#61a0a8", "#d48265",
      "#749f83", "#ca8622", "#bda29a", "#6e7074", "#546570", "#c4ccd3",
      "#6f8edc", "#a7d88a", "#ffd976", "#f27c7c", "#8ad0ea", "#55b487",
      "#fd9b70", "#b07ac6", "#ef91d2", "#405f70", "#79b3ba", "#dd987d",
      "#8db796", "#d59a45", "#cbb8af", "#85878b", "#6d7f8d", "#d3d9df"
    ];

    const chart = echarts.init(document.getElementById("network-chart"), null, { renderer: "canvas" });
    let networkData = null;
    let baseNodes = [];
    let currentNodes = [];
    let communityTargets = new Map();
    let simulation = null;
    let showLabels = true;

    chart.showLoading({
      text: "Loading network data...",
      color: "#5470c6",
      textColor: "#1f2937",
      maskColor: "rgba(255,255,255,0.85)"
    });

    fetch(DATA_FILE)
      .then(response => {
        if (!response.ok) throw new Error("Cannot load " + DATA_FILE);
        return response.json();
      })
      .then(data => {
        networkData = data;
        window.epbHybridNetworkData = networkData;
        assignCommunityColors(networkData);
        baseNodes = deepCopy(networkData.nodes).map(node => {
          const baseFont = node.label && node.label.fontSize ? node.label.fontSize : 12;
          node._baseSymbolSize = node.symbolSize || 8;
          node._baseFontSize = baseFont;
          return node;
        });
        currentNodes = deepCopy(baseNodes);
        populateCommunitySelect();
        chart.hideLoading();
        runHybridLayout(true);
      })
      .catch(error => {
        chart.hideLoading();
        const message = "Network data was not loaded. Open this HTML through a local server such as VS Code Live Server. " + error.message;
        document.getElementById("network-chart").innerHTML = "<div style='padding:24px;font:16px Arial;color:#b91c1c;'>" + message + "</div>";
      });

    window.addEventListener("resize", () => {
      chart.resize();
      if (networkData) runHybridLayout(true);
    });
    chart.on("mouseup", () => {
      currentNodes = captureCurrentNodes();
    });

    function deepCopy(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function seedRandom(seed) {
      let value = seed % 2147483647;
      if (value <= 0) value += 2147483646;
      return function() {
        value = value * 16807 % 2147483647;
        return (value - 1) / 2147483646;
      };
    }

    function assignCommunityColors(data) {
      data.categories.forEach((category, index) => {
        category.itemStyle = Object.assign({}, category.itemStyle || {}, {
          color: ECHARTS_STYLE_PALETTE[index % ECHARTS_STYLE_PALETTE.length]
        });
      });
      data.nodes.forEach(node => {
        const category = data.categories[node.category] || {};
        const color = category.itemStyle ? category.itemStyle.color : ECHARTS_STYLE_PALETTE[0];
        node.itemStyle = Object.assign({}, node.itemStyle || {}, {
          color: color,
          borderColor: "#ffffff",
          borderWidth: 1
        });
      });
    }

    function populateCommunitySelect() {
      const select = document.getElementById("communitySelect");
      select.innerHTML = "";
      networkData.categories.forEach((category, index) => {
        const count = networkData.nodes.filter(node => node.category === index).length;
        if (count === 0) return;
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = category.name + " (" + count + ")";
        select.appendChild(option);
      });
    }

    function selectedCommunityId() {
      return Number(document.getElementById("communitySelect").value);
    }

    function runHybridLayout(resetPositions) {
      if (!networkData || !window.d3) return;
      stopSimulation();
      buildCommunityTargets();
      currentNodes = resetPositions ? deepCopy(baseNodes) : captureCurrentNodes();
      applySizeScales(false);
      if (resetPositions || !hasFinitePositions(currentNodes)) {
        initializeNodePositions();
      }
      currentNodes.forEach(node => {
        delete node.fx;
        delete node.fy;
        node.fixed = true;
      });

      const forceLinks = networkData.links.map(link => ({
        source: String(link.source),
        target: String(link.target),
        value: Number(link.value || link.weight || 1)
      }));

      simulation = d3.forceSimulation(currentNodes)
        .force("link", d3.forceLink(forceLinks)
          .id(node => String(node.id))
          .distance(link => 58 + 34 / Math.max(0.35, Math.min(3, link.value || 1)))
          .strength(link => Math.min(0.20, 0.045 + 0.045 * Math.min(2.5, link.value || 1))))
        .force("charge", d3.forceManyBody()
          .strength(node => -22 - nodeRadius(node) * 1.4)
          .distanceMax(310))
        .force("collide", d3.forceCollide(node => nodeRadius(node) + 4)
          .iterations(2)
          .strength(0.92))
        .force("x", d3.forceX(node => targetFor(node).x)
          .strength(Number(document.getElementById("anchorPull").value)))
        .force("y", d3.forceY(node => targetFor(node).y)
          .strength(Number(document.getElementById("anchorPull").value)))
        .force("boundary", boundaryForce(34))
        .alpha(1)
        .alphaDecay(0.018)
        .velocityDecay(0.34)
        .stop();
      if (simulation.randomSource) simulation.randomSource(seedRandom(42));

      tickSimulation(340, 10);
    }

    function buildCommunityTargets() {
      const grouped = groupNodesByCommunity(baseNodes);
      const communities = Array.from(grouped.entries()).map(([category, nodes]) => ({
        category,
        count: nodes.length,
        r: Math.max(44, Math.sqrt(nodes.length) * 18 + 22)
      })).sort((a, b) => b.count - a.count || a.category - b.category);

      const width = Math.max(700, chart.getWidth());
      const height = Math.max(480, chart.getHeight());
      const fill = Number(document.getElementById("fillScale").value);
      const cx = width / 2;
      const cy = height / 2;
      const rx = width * 0.46 * fill;
      const ry = height * 0.43 * fill;
      const goldenAngle = 2.399963229728653;
      communities.forEach((community, index) => {
        const rank = communities.length === 1 ? 0 : index / (communities.length - 1);
        const radius = Math.sqrt(rank);
        const angle = index * goldenAngle + 0.25;
        community.x = cx + Math.cos(angle) * rx * radius;
        community.y = cy + Math.sin(angle) * ry * radius;
        community.x0 = community.x;
        community.y0 = community.y;
      });

      const targetSimulation = d3.forceSimulation(communities)
        .force("collide", d3.forceCollide(d => d.r + 16).iterations(3).strength(0.95))
        .force("x", d3.forceX(d => d.x0).strength(0.20))
        .force("y", d3.forceY(d => d.y0).strength(0.20))
        .force("boundary", boundaryForce(62))
        .alpha(1)
        .alphaDecay(0.025)
        .stop();
      if (targetSimulation.randomSource) targetSimulation.randomSource(seedRandom(84));
      for (let i = 0; i < 220; i += 1) targetSimulation.tick();

      communityTargets = new Map();
      communities.forEach(community => {
        communityTargets.set(community.category, { x: community.x, y: community.y, r: community.r });
      });
    }

    function groupNodesByCommunity(nodes) {
      const grouped = new Map();
      nodes.forEach(node => {
        if (!grouped.has(node.category)) grouped.set(node.category, []);
        grouped.get(node.category).push(node);
      });
      return grouped;
    }

    function initializeNodePositions() {
      const grouped = groupNodesByCommunity(currentNodes);
      grouped.forEach((nodes, category) => {
        const target = communityTargets.get(category) || { x: chart.getWidth() / 2, y: chart.getHeight() / 2, r: 80 };
        const sorted = nodes.slice().sort((a, b) =>
          (b.weighted_degree || 0) - (a.weighted_degree || 0) || String(a.name).localeCompare(String(b.name))
        );
        sorted.forEach((node, index) => {
          if (sorted.length === 1) {
            node.x = target.x;
            node.y = target.y;
          } else {
            const t = index / Math.max(sorted.length - 1, 1);
            const radius = Math.sqrt(t) * target.r * 0.74;
            const angle = index * 2.399963229728653;
            node.x = target.x + Math.cos(angle) * radius;
            node.y = target.y + Math.sin(angle) * radius * 0.72;
          }
          node.fixed = true;
        });
      });
    }

    function hasFinitePositions(nodes) {
      return nodes.length > 0 && nodes.every(node => Number.isFinite(node.x) && Number.isFinite(node.y));
    }

    function targetFor(node) {
      return communityTargets.get(node.category) || { x: chart.getWidth() / 2, y: chart.getHeight() / 2 };
    }

    function boundaryForce(padding) {
      let nodes = [];
      function force(alpha) {
        const minX = padding;
        const minY = padding;
        const maxX = Math.max(minX + 1, chart.getWidth() - padding);
        const maxY = Math.max(minY + 1, chart.getHeight() - padding);
        nodes.forEach(node => {
          const radius = node.r || nodeRadius(node);
          if (node.x - radius < minX) node.vx += (minX - (node.x - radius)) * 0.08 * alpha;
          if (node.x + radius > maxX) node.vx -= ((node.x + radius) - maxX) * 0.08 * alpha;
          if (node.y - radius < minY) node.vy += (minY - (node.y - radius)) * 0.08 * alpha;
          if (node.y + radius > maxY) node.vy -= ((node.y + radius) - maxY) * 0.08 * alpha;
        });
      }
      force.initialize = function(_) {
        nodes = _;
      };
      return force;
    }

    function tickSimulation(remainingTicks, ticksPerFrame) {
      if (!simulation) return;
      const ticks = Math.min(ticksPerFrame, remainingTicks);
      for (let i = 0; i < ticks; i += 1) simulation.tick();
      currentNodes.forEach(node => {
        node.fixed = true;
      });
      renderNetwork(false);
      if (remainingTicks > ticks) {
        requestAnimationFrame(() => tickSimulation(remainingTicks - ticks, ticksPerFrame));
      } else {
        stopSimulation(false);
        renderNetwork(false);
      }
    }

    function stopSimulation(renderAfterStop) {
      if (simulation) {
        simulation.stop();
        simulation = null;
      }
      if (renderAfterStop !== false && networkData) renderNetwork(false);
    }

    function nodeRadius(node) {
      return (node.symbolSize || node._baseSymbolSize || 8) * 0.62 + labelRadius(node);
    }

    function labelRadius(node) {
      const label = node.label || {};
      const fontSize = label.fontSize || node._baseFontSize || 12;
      const text = node.name || "";
      if (!showLabels || label.show === false || text.length === 0) return 0;
      return Math.min(105, text.length * fontSize * 0.28);
    }

    function captureCurrentNodes() {
      const series = chart.getModel().getSeriesByIndex(0);
      if (!series) return currentNodes;
      const data = series.getData();
      return currentNodes.map((node, index) => {
        const next = Object.assign({}, node);
        const layout = data.getItemLayout(index);
        if (layout && Number.isFinite(layout[0]) && Number.isFinite(layout[1])) {
          next.x = layout[0];
          next.y = layout[1];
        }
        next.fixed = true;
        return next;
      });
    }

    function renderNetwork(reset) {
      chart.setOption({
        backgroundColor: "#ffffff",
        tooltip: {
          trigger: "item",
          formatter: tooltipText,
          backgroundColor: "rgba(255,255,255,0.96)",
          borderColor: "#94a3b8",
          borderWidth: 1,
          textStyle: { color: "#1f2937", fontSize: 12 }
        },
        legend: [{ data: networkData.categories.map(item => item.name), show: false }],
        series: [{
          id: SERIES_ID,
          name: "EPB Hybrid Force",
          type: "graph",
          symbol: "circle",
          layout: "none",
          data: nodesWithAspectAnchors(),
          links: networkData.links,
          categories: networkData.categories,
          left: 0,
          top: 0,
          right: 0,
          bottom: 0,
          roam: true,
          draggable: true,
          focusNodeAdjacency: true,
          animation: false,
          itemStyle: {
            borderColor: "#ffffff",
            borderWidth: 1
          },
          lineStyle: {
            color: "source",
            curveness: 0.04,
            opacity: 0.28
          },
          label: {
            show: showLabels,
            position: "right",
            formatter: "{b}",
            color: "#1f2937",
            fontWeight: "bold",
            textBorderColor: "#ffffff",
            textBorderWidth: 3
          },
          labelLayout: {
            hideOverlap: false
          },
          scaleLimit: {
            min: 0.02,
            max: 20
          },
          emphasis: {
            focus: "adjacency",
            lineStyle: {
              opacity: 0.85,
              width: 4
            }
          }
        }]
      }, Boolean(reset));
    }

    function nodesWithAspectAnchors() {
      const width = Math.max(1, chart.getWidth());
      const height = Math.max(1, chart.getHeight());
      const anchorStyle = { opacity: 0, color: "rgba(0,0,0,0)", borderWidth: 0 };
      const anchors = [
        {
          id: "__aspect_anchor_top_left",
          name: "",
          x: 0,
          y: 0,
          fixed: true,
          symbol: "circle",
          symbolSize: 0,
          category: 0,
          value: 0,
          silent: true,
          tooltip: { show: false },
          label: { show: false },
          itemStyle: anchorStyle
        },
        {
          id: "__aspect_anchor_bottom_right",
          name: "",
          x: width,
          y: height,
          fixed: true,
          symbol: "circle",
          symbolSize: 0,
          category: 0,
          value: 0,
          silent: true,
          tooltip: { show: false },
          label: { show: false },
          itemStyle: anchorStyle
        }
      ];
      return currentNodes.map(node => Object.assign({ symbol: "circle" }, node)).concat(anchors);
    }

    function tooltipText(params) {
      if (params.dataType === "node") {
        const category = networkData.categories[params.data.category] || { name: "Unknown" };
        return [
          "<strong>" + params.data.full_name + "</strong>",
          "Papers: " + params.data.papers,
          "Weighted Degree: " + params.data.weighted_degree,
          "Community: " + category.name
        ].join("<br>");
      }
      if (params.dataType === "edge") {
        return [
          "<strong>Collaboration</strong>",
          "Weight: " + params.data.value,
          "Between: " + params.data.source + " - " + params.data.target
        ].join("<br>");
      }
      return "";
    }

    function selectedCommunityNodes() {
      const category = selectedCommunityId();
      return currentNodes.filter(node => node.category === category);
    }

    function moveSelectedCommunity(corner) {
      stopSimulation(false);
      currentNodes = captureCurrentNodes();
      const nodes = selectedCommunityNodes();
      if (nodes.length === 0) return;
      const box = boundingBox(nodes);
      const targets = {
        "top-left": [chart.getWidth() * 0.16, chart.getHeight() * 0.18],
        "top-right": [chart.getWidth() * 0.84, chart.getHeight() * 0.18],
        "bottom-left": [chart.getWidth() * 0.16, chart.getHeight() * 0.82],
        "bottom-right": [chart.getWidth() * 0.84, chart.getHeight() * 0.82]
      };
      const target = targets[corner];
      translateCommunity(selectedCommunityId(), target[0] - box.cx, target[1] - box.cy);
    }

    function nudgeSelectedCommunity(dx, dy) {
      stopSimulation(false);
      currentNodes = captureCurrentNodes();
      const step = Number(document.getElementById("nudgeStep").value);
      translateCommunity(selectedCommunityId(), dx * step, dy * step);
    }

    function translateCommunity(category, dx, dy) {
      currentNodes = currentNodes.map(node => {
        if (node.category !== category) return node;
        return Object.assign({}, node, {
          x: node.x + dx,
          y: node.y + dy,
          fixed: true
        });
      });
      renderNetwork(false);
    }

    function spreadSelectedCommunity() {
      stopSimulation(false);
      currentNodes = captureCurrentNodes();
      scaleCommunity(selectedCommunityId(), Number(document.getElementById("spreadStep").value));
    }

    function tightenSelectedCommunity() {
      stopSimulation(false);
      currentNodes = captureCurrentNodes();
      scaleCommunity(selectedCommunityId(), 1 / Number(document.getElementById("spreadStep").value));
    }

    function scaleCommunity(category, factor) {
      const nodes = currentNodes.filter(node => node.category === category);
      if (nodes.length === 0) return;
      const box = boundingBox(nodes);
      currentNodes = currentNodes.map(node => {
        if (node.category !== category) return node;
        return Object.assign({}, node, {
          x: box.cx + (node.x - box.cx) * factor,
          y: box.cy + (node.y - box.cy) * factor,
          fixed: true
        });
      });
      renderNetwork(false);
    }

    function boundingBox(nodes) {
      const xs = nodes.map(node => node.x).filter(Number.isFinite);
      const ys = nodes.map(node => node.y).filter(Number.isFinite);
      const minX = Math.min.apply(null, xs);
      const maxX = Math.max.apply(null, xs);
      const minY = Math.min.apply(null, ys);
      const maxY = Math.max.apply(null, ys);
      return {
        minX,
        maxX,
        minY,
        maxY,
        cx: (minX + maxX) / 2,
        cy: (minY + maxY) / 2
      };
    }

    function resetView() {
      chart.dispatchAction({ type: "restore" });
    }

    function toggleLabels() {
      showLabels = !showLabels;
      currentNodes = captureCurrentNodes();
      applySizeScales(false);
      renderNetwork(false);
    }

    function updateSizes() {
      stopSimulation(false);
      currentNodes = captureCurrentNodes();
      applySizeScales(false);
      renderNetwork(false);
    }

    function applySizeScales(resetPositions) {
      const nodeScale = Number(document.getElementById("nodeScale").value);
      const labelScale = Number(document.getElementById("labelScale").value);
      const positions = new Map(currentNodes.map(node => [node.id, { x: node.x, y: node.y }]));
      currentNodes = currentNodes.map(node => {
        const next = Object.assign({}, node);
        const base = baseNodes.find(item => item.id === node.id) || node;
        next.symbolSize = Math.max(2, Math.round((base._baseSymbolSize || base.symbolSize || 8) * nodeScale));
        const baseLabel = base.label || {};
        next.label = Object.assign({}, baseLabel, {
          show: showLabels,
          fontSize: Math.max(6, Math.round((base._baseFontSize || baseLabel.fontSize || 12) * labelScale))
        });
        if (!resetPositions && positions.has(node.id)) {
          next.x = positions.get(node.id).x;
          next.y = positions.get(node.id).y;
        }
        next.fixed = true;
        return next;
      });
    }

    function saveAsImage() {
      const url = chart.getDataURL({
        type: "png",
        pixelRatio: 3,
        backgroundColor: "#ffffff"
      });
      const link = document.createElement("a");
      link.download = "epb-hybrid-force-network.png";
      link.href = url;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  </script>
</body>
</html>
"""
    html_path.write_text(html_text.replace("__DATA_FILE__", data_file), encoding="utf-8")


def write_hybrid_force_svg_network_html(html_path: Path, data_file: str):
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EPB Hybrid Force SVG Network</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #ffffff;
      color: #1f2937;
      font-family: Arial, Helvetica, sans-serif;
    }
    .toolbar {
      height: 52px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      box-sizing: border-box;
      border-bottom: 1px solid #e5e7eb;
      background: #ffffff;
      white-space: nowrap;
      overflow-x: auto;
    }
    button, select, input {
      font: inherit;
    }
    button {
      height: 34px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #f8fafc;
      color: #111827;
      padding: 0 10px;
      cursor: pointer;
    }
    button:hover {
      background: #eef2f7;
    }
    select {
      height: 34px;
      min-width: 160px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: #111827;
      padding: 0 8px;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #475569;
      font-size: 13px;
    }
    input[type="range"] {
      width: 88px;
    }
    #network-chart {
      width: 100vw;
      height: calc(100vh - 52px);
      background: #ffffff;
    }
    svg {
      display: block;
      width: 100%;
      height: 100%;
      background: #ffffff;
    }
    .network-link {
      fill: none;
      stroke-linecap: round;
      pointer-events: none;
    }
    .network-node {
      stroke: #ffffff;
      stroke-width: 1.1px;
      cursor: grab;
    }
    .network-node:active {
      cursor: grabbing;
    }
    .network-label {
      fill: #1f2937;
      font-weight: 700;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 4px;
      stroke-linejoin: round;
      pointer-events: none;
      dominant-baseline: middle;
    }
    .tooltip {
      position: fixed;
      z-index: 10;
      max-width: 300px;
      padding: 10px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.16);
      color: #1f2937;
      font-size: 12px;
      line-height: 1.45;
      pointer-events: none;
      opacity: 0;
      transform: translate(10px, 10px);
    }
    .tooltip strong {
      display: block;
      margin-bottom: 4px;
      font-size: 13px;
      color: #111827;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button onclick="runHybridLayout(true)">Hybrid Pack</button>
    <button onclick="runHybridLayout(false)">Refine Force</button>
    <button onclick="stopSimulation()">Stop</button>
    <button onclick="resetView()">Reset View</button>
    <button onclick="toggleLabels()">Toggle Labels</button>
    <button onclick="saveAsPng()">Save PNG</button>
    <select id="communitySelect"></select>
    <button onclick="moveSelectedCommunity('top-left')">Top Left</button>
    <button onclick="moveSelectedCommunity('top-right')">Top Right</button>
    <button onclick="moveSelectedCommunity('bottom-left')">Bottom Left</button>
    <button onclick="moveSelectedCommunity('bottom-right')">Bottom Right</button>
    <button onclick="nudgeSelectedCommunity(-1, 0)">Left</button>
    <button onclick="nudgeSelectedCommunity(1, 0)">Right</button>
    <button onclick="nudgeSelectedCommunity(0, -1)">Up</button>
    <button onclick="nudgeSelectedCommunity(0, 1)">Down</button>
    <button onclick="spreadSelectedCommunity()">Spread</button>
    <button onclick="tightenSelectedCommunity()">Tighten</button>
    <label>Fill <input id="fillScale" type="range" min="0.65" max="1.08" step="0.03" value="0.94"></label>
    <label>Pull <input id="anchorPull" type="range" min="0.015" max="0.11" step="0.005" value="0.055"></label>
    <label>Step <input id="nudgeStep" type="range" min="20" max="260" step="20" value="100"></label>
    <label>Spread <input id="spreadStep" type="range" min="1.05" max="2.50" step="0.05" value="1.25"></label>
    <label>Node <input id="nodeScale" type="range" min="0.5" max="2.6" step="0.1" value="1.0" oninput="updateSizes()"></label>
    <label>Label <input id="labelScale" type="range" min="0.5" max="3.2" step="0.1" value="1.0" oninput="updateSizes()"></label>
  </div>
  <div id="network-chart"></div>
  <div id="tooltip" class="tooltip"></div>

  <script>
    const DATA_FILE = "__DATA_FILE__";
    const ECHARTS_STYLE_PALETTE = [
      "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272",
      "#fc8452", "#9a60b4", "#ea7ccc", "#2f4554", "#61a0a8", "#d48265",
      "#749f83", "#ca8622", "#bda29a", "#6e7074", "#546570", "#c4ccd3",
      "#6f8edc", "#a7d88a", "#ffd976", "#f27c7c", "#8ad0ea", "#55b487",
      "#fd9b70", "#b07ac6", "#ef91d2", "#405f70", "#79b3ba", "#dd987d",
      "#8db796", "#d59a45", "#cbb8af", "#85878b", "#6d7f8d", "#d3d9df"
    ];

    const container = document.getElementById("network-chart");
    const svg = d3.select(container).append("svg").attr("xmlns", "http://www.w3.org/2000/svg");
    const viewport = svg.append("g").attr("class", "viewport");
    const linkLayer = viewport.append("g").attr("class", "links");
    const nodeLayer = viewport.append("g").attr("class", "nodes");
    const labelLayer = viewport.append("g").attr("class", "labels");
    const zoom = d3.zoom().scaleExtent([0.08, 14]).on("zoom", event => {
      viewport.attr("transform", event.transform);
    });
    svg.call(zoom);
    const tooltip = d3.select("#tooltip");

    let width = 1;
    let height = 1;
    let networkData = null;
    let baseNodes = [];
    let currentNodes = [];
    let currentLinks = [];
    let communityTargets = new Map();
    let simulation = null;
    let showLabels = true;
    let neighborMap = new Map();

    fetch(DATA_FILE)
      .then(response => {
        if (!response.ok) throw new Error("Cannot load " + DATA_FILE);
        return response.json();
      })
      .then(data => {
        networkData = data;
        window.epbHybridSvgData = networkData;
        assignCommunityColors(networkData);
        baseNodes = networkData.nodes.map(node => {
          const next = Object.assign({}, node);
          const baseFont = next.label && next.label.fontSize ? next.label.fontSize : 12;
          next._baseSymbolSize = next.symbolSize || 8;
          next._baseFontSize = baseFont;
          next.color = colorForCategory(next.category);
          return next;
        });
        currentLinks = networkData.links.map(link => ({
          source: String(link.source),
          target: String(link.target),
          value: Number(link.value || link.weight || 1)
        }));
        buildNeighborMap();
        populateCommunitySelect();
        resizeSvg();
        runHybridLayout(true);
      })
      .catch(error => {
        container.innerHTML = "<div style='padding:24px;font:16px Arial;color:#b91c1c;'>" + error.message + "</div>";
      });

    window.addEventListener("resize", () => {
      resizeSvg();
      if (networkData) runHybridLayout(true);
    });

    function resizeSvg() {
      width = Math.max(1, container.clientWidth);
      height = Math.max(1, container.clientHeight);
      svg.attr("viewBox", "0 0 " + width + " " + height);
    }

    function seedRandom(seed) {
      let value = seed % 2147483647;
      if (value <= 0) value += 2147483646;
      return function() {
        value = value * 16807 % 2147483647;
        return (value - 1) / 2147483646;
      };
    }

    function assignCommunityColors(data) {
      data.categories.forEach((category, index) => {
        category.color = ECHARTS_STYLE_PALETTE[index % ECHARTS_STYLE_PALETTE.length];
      });
    }

    function colorForCategory(category) {
      const item = networkData.categories[category];
      return item && item.color ? item.color : ECHARTS_STYLE_PALETTE[0];
    }

    function populateCommunitySelect() {
      const select = document.getElementById("communitySelect");
      select.innerHTML = "";
      networkData.categories.forEach((category, index) => {
        const count = networkData.nodes.filter(node => node.category === index).length;
        if (count === 0) return;
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = category.name + " (" + count + ")";
        select.appendChild(option);
      });
    }

    function selectedCommunityId() {
      return Number(document.getElementById("communitySelect").value);
    }

    function buildNeighborMap() {
      neighborMap = new Map();
      baseNodes.forEach(node => neighborMap.set(String(node.id), new Set()));
      currentLinks.forEach(link => {
        const source = String(link.source);
        const target = String(link.target);
        if (!neighborMap.has(source)) neighborMap.set(source, new Set());
        if (!neighborMap.has(target)) neighborMap.set(target, new Set());
        neighborMap.get(source).add(target);
        neighborMap.get(target).add(source);
      });
    }

    function runHybridLayout(resetPositions) {
      if (!networkData || !window.d3) return;
      stopSimulation(false);
      buildCommunityTargets();
      currentNodes = resetPositions ? baseNodes.map(node => Object.assign({}, node)) : currentNodes.map(node => Object.assign({}, node));
      applySizeScales(false);
      if (resetPositions || !hasFinitePositions(currentNodes)) {
        initializeNodePositions();
      }

      simulation = d3.forceSimulation(currentNodes)
        .force("link", d3.forceLink(currentLinks)
          .id(node => String(node.id))
          .distance(link => 58 + 34 / Math.max(0.35, Math.min(3, link.value || 1)))
          .strength(link => Math.min(0.20, 0.045 + 0.045 * Math.min(2.5, link.value || 1))))
        .force("charge", d3.forceManyBody()
          .strength(node => -22 - nodeRadius(node) * 1.4)
          .distanceMax(310))
        .force("collide", d3.forceCollide(node => nodeRadius(node) + 4)
          .iterations(2)
          .strength(0.92))
        .force("x", d3.forceX(node => targetFor(node).x)
          .strength(Number(document.getElementById("anchorPull").value)))
        .force("y", d3.forceY(node => targetFor(node).y)
          .strength(Number(document.getElementById("anchorPull").value)))
        .force("boundary", boundaryForce(34))
        .alpha(1)
        .alphaDecay(0.018)
        .velocityDecay(0.34)
        .on("tick", renderPositions)
        .on("end", renderPositions);
      if (simulation.randomSource) simulation.randomSource(seedRandom(42));
      renderNetwork();
    }

    function buildCommunityTargets() {
      const grouped = groupNodesByCommunity(baseNodes);
      const communities = Array.from(grouped.entries()).map(([category, nodes]) => ({
        category,
        count: nodes.length,
        r: Math.max(44, Math.sqrt(nodes.length) * 18 + 22)
      })).sort((a, b) => b.count - a.count || a.category - b.category);

      const fill = Number(document.getElementById("fillScale").value);
      const cx = width / 2;
      const cy = height / 2;
      const rx = width * 0.46 * fill;
      const ry = height * 0.43 * fill;
      const goldenAngle = 2.399963229728653;
      communities.forEach((community, index) => {
        const rank = communities.length === 1 ? 0 : index / (communities.length - 1);
        const radius = Math.sqrt(rank);
        const angle = index * goldenAngle + 0.25;
        community.x = cx + Math.cos(angle) * rx * radius;
        community.y = cy + Math.sin(angle) * ry * radius;
        community.x0 = community.x;
        community.y0 = community.y;
      });

      const targetSimulation = d3.forceSimulation(communities)
        .force("collide", d3.forceCollide(d => d.r + 16).iterations(3).strength(0.95))
        .force("x", d3.forceX(d => d.x0).strength(0.20))
        .force("y", d3.forceY(d => d.y0).strength(0.20))
        .force("boundary", boundaryForce(62))
        .alpha(1)
        .alphaDecay(0.025)
        .stop();
      if (targetSimulation.randomSource) targetSimulation.randomSource(seedRandom(84));
      for (let i = 0; i < 220; i += 1) targetSimulation.tick();

      communityTargets = new Map();
      communities.forEach(community => {
        communityTargets.set(community.category, { x: community.x, y: community.y, r: community.r });
      });
    }

    function groupNodesByCommunity(nodes) {
      const grouped = new Map();
      nodes.forEach(node => {
        if (!grouped.has(node.category)) grouped.set(node.category, []);
        grouped.get(node.category).push(node);
      });
      return grouped;
    }

    function initializeNodePositions() {
      const grouped = groupNodesByCommunity(currentNodes);
      grouped.forEach((nodes, category) => {
        const target = communityTargets.get(category) || { x: width / 2, y: height / 2, r: 80 };
        const sorted = nodes.slice().sort((a, b) =>
          (b.weighted_degree || 0) - (a.weighted_degree || 0) || String(a.name).localeCompare(String(b.name))
        );
        sorted.forEach((node, index) => {
          if (sorted.length === 1) {
            node.x = target.x;
            node.y = target.y;
          } else {
            const t = index / Math.max(sorted.length - 1, 1);
            const radius = Math.sqrt(t) * target.r * 0.74;
            const angle = index * 2.399963229728653;
            node.x = target.x + Math.cos(angle) * radius;
            node.y = target.y + Math.sin(angle) * radius * 0.72;
          }
        });
      });
    }

    function hasFinitePositions(nodes) {
      return nodes.length > 0 && nodes.every(node => Number.isFinite(node.x) && Number.isFinite(node.y));
    }

    function targetFor(node) {
      return communityTargets.get(node.category) || { x: width / 2, y: height / 2 };
    }

    function boundaryForce(padding) {
      let nodes = [];
      function force(alpha) {
        const minX = padding;
        const minY = padding;
        const maxX = Math.max(minX + 1, width - padding);
        const maxY = Math.max(minY + 1, height - padding);
        nodes.forEach(node => {
          const radius = node.r || nodeRadius(node);
          if (node.x - radius < minX) node.vx += (minX - (node.x - radius)) * 0.08 * alpha;
          if (node.x + radius > maxX) node.vx -= ((node.x + radius) - maxX) * 0.08 * alpha;
          if (node.y - radius < minY) node.vy += (minY - (node.y - radius)) * 0.08 * alpha;
          if (node.y + radius > maxY) node.vy -= ((node.y + radius) - maxY) * 0.08 * alpha;
        });
      }
      force.initialize = function(_) {
        nodes = _;
      };
      return force;
    }

    function stopSimulation(renderAfterStop) {
      if (simulation) {
        simulation.stop();
        simulation = null;
      }
      if (renderAfterStop !== false) renderPositions();
    }

    function nodeRadius(node) {
      return (node.symbolSize || node._baseSymbolSize || 8) * 0.5 + labelRadius(node) * 0.24;
    }

    function labelRadius(node) {
      const label = node.label || {};
      const fontSize = label.fontSize || node._baseFontSize || 12;
      const text = node.name || "";
      if (!showLabels || label.show === false || text.length === 0) return 0;
      return Math.min(96, text.length * fontSize * 0.30);
    }

    function renderNetwork() {
      const links = linkLayer.selectAll("line")
        .data(currentLinks, d => String(d.source.id || d.source) + "-" + String(d.target.id || d.target));
      links.exit().remove();
      links.enter()
        .append("line")
        .attr("class", "network-link")
        .merge(links)
        .attr("stroke", d => sourceColor(d))
        .attr("stroke-opacity", 0.30)
        .attr("stroke-width", d => 0.65 + Math.min(2.6, d.value || 1) * 0.55);

      const nodes = nodeLayer.selectAll("circle")
        .data(currentNodes, d => d.id);
      nodes.exit().remove();
      nodes.enter()
        .append("circle")
        .attr("class", "network-node")
        .on("mouseenter", showNodeTooltip)
        .on("mousemove", moveTooltip)
        .on("mouseleave", hideNodeTooltip)
        .call(dragBehavior())
        .merge(nodes)
        .attr("r", d => Math.max(2, (d.symbolSize || 8) / 2))
        .attr("fill", d => d.color || colorForCategory(d.category));

      const labels = labelLayer.selectAll("text")
        .data(showLabels ? currentNodes : [], d => d.id);
      labels.exit().remove();
      labels.enter()
        .append("text")
        .attr("class", "network-label")
        .merge(labels)
        .text(d => d.name || "")
        .style("font-size", d => ((d.label && d.label.fontSize) || d._baseFontSize || 12) + "px");

      renderPositions();
    }

    function renderPositions() {
      linkLayer.selectAll("line")
        .attr("x1", d => nodeByEndpoint(d.source).x)
        .attr("y1", d => nodeByEndpoint(d.source).y)
        .attr("x2", d => nodeByEndpoint(d.target).x)
        .attr("y2", d => nodeByEndpoint(d.target).y);
      nodeLayer.selectAll("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
      labelLayer.selectAll("text")
        .attr("x", d => d.x + Math.max(4, (d.symbolSize || 8) / 2 + 4))
        .attr("y", d => d.y + 0.5);
    }

    function nodeByEndpoint(endpoint) {
      if (typeof endpoint === "object") return endpoint;
      return currentNodes.find(node => String(node.id) === String(endpoint)) || { x: 0, y: 0 };
    }

    function sourceColor(link) {
      const node = nodeByEndpoint(link.source);
      return node.color || colorForCategory(node.category || 0);
    }

    function linkSourceId(link) {
      return typeof link.source === "object" ? String(link.source.id) : String(link.source);
    }

    function linkTargetId(link) {
      return typeof link.target === "object" ? String(link.target.id) : String(link.target);
    }

    function showNodeTooltip(event, node) {
      const nodeId = String(node.id);
      const neighbors = neighborMap.get(nodeId) || new Set();
      const category = networkData.categories[node.category] || { name: "Unknown" };
      const degree = neighbors.size;
      tooltip
        .style("opacity", 1)
        .html(
          "<strong>" + escapeHtml(node.full_name || node.name || node.id) + "</strong>" +
          "Papers: " + safeValue(node.papers) + "<br>" +
          "Weighted degree: " + safeValue(node.weighted_degree) + "<br>" +
          "Collaborators: " + degree + "<br>" +
          "Community: " + escapeHtml(category.name || "Unknown")
        );
      moveTooltip(event);
      highlightNeighborhood(nodeId, neighbors);
    }

    function moveTooltip(event) {
      tooltip
        .style("left", Math.min(window.innerWidth - 320, event.clientX + 14) + "px")
        .style("top", Math.min(window.innerHeight - 130, event.clientY + 14) + "px");
    }

    function hideNodeTooltip() {
      tooltip.style("opacity", 0);
      clearHighlight();
    }

    function highlightNeighborhood(nodeId, neighbors) {
      nodeLayer.selectAll("circle")
        .attr("opacity", d => String(d.id) === nodeId || neighbors.has(String(d.id)) ? 1 : 0.22)
        .attr("stroke-width", d => String(d.id) === nodeId ? 2.8 : 1.1);
      labelLayer.selectAll("text")
        .attr("opacity", d => String(d.id) === nodeId || neighbors.has(String(d.id)) ? 1 : 0.18);
      linkLayer.selectAll("line")
        .attr("stroke-opacity", d => {
          const source = linkSourceId(d);
          const target = linkTargetId(d);
          return source === nodeId || target === nodeId ? 0.88 : 0.08;
        })
        .attr("stroke-width", d => {
          const source = linkSourceId(d);
          const target = linkTargetId(d);
          return source === nodeId || target === nodeId ? 2.4 : 0.65 + Math.min(2.6, d.value || 1) * 0.35;
        });
    }

    function clearHighlight() {
      nodeLayer.selectAll("circle")
        .attr("opacity", 1)
        .attr("stroke-width", 1.1);
      labelLayer.selectAll("text")
        .attr("opacity", 1);
      linkLayer.selectAll("line")
        .attr("stroke-opacity", 0.30)
        .attr("stroke-width", d => 0.65 + Math.min(2.6, d.value || 1) * 0.55);
    }

    function safeValue(value) {
      return value === undefined || value === null || value === "" ? "NA" : value;
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function dragBehavior() {
      return d3.drag()
        .on("start", (event, d) => {
          if (simulation) simulation.alphaTarget(0.15).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
          d.x = event.x;
          d.y = event.y;
          renderPositions();
        })
        .on("end", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
          d.x = event.x;
          d.y = event.y;
          if (simulation) simulation.alphaTarget(0);
          renderPositions();
        });
    }

    function selectedCommunityNodes() {
      const category = selectedCommunityId();
      return currentNodes.filter(node => node.category === category);
    }

    function moveSelectedCommunity(corner) {
      stopSimulation(false);
      const nodes = selectedCommunityNodes();
      if (nodes.length === 0) return;
      const box = boundingBox(nodes);
      const targets = {
        "top-left": [width * 0.16, height * 0.18],
        "top-right": [width * 0.84, height * 0.18],
        "bottom-left": [width * 0.16, height * 0.82],
        "bottom-right": [width * 0.84, height * 0.82]
      };
      const target = targets[corner];
      translateCommunity(selectedCommunityId(), target[0] - box.cx, target[1] - box.cy);
    }

    function nudgeSelectedCommunity(dx, dy) {
      stopSimulation(false);
      const step = Number(document.getElementById("nudgeStep").value);
      translateCommunity(selectedCommunityId(), dx * step, dy * step);
    }

    function translateCommunity(category, dx, dy) {
      currentNodes.forEach(node => {
        if (node.category !== category) return;
        node.x += dx;
        node.y += dy;
        node.fx = node.x;
        node.fy = node.y;
      });
      renderPositions();
    }

    function spreadSelectedCommunity() {
      stopSimulation(false);
      scaleCommunity(selectedCommunityId(), Number(document.getElementById("spreadStep").value));
    }

    function tightenSelectedCommunity() {
      stopSimulation(false);
      scaleCommunity(selectedCommunityId(), 1 / Number(document.getElementById("spreadStep").value));
    }

    function scaleCommunity(category, factor) {
      const nodes = selectedCommunityNodes();
      if (nodes.length === 0) return;
      const box = boundingBox(nodes);
      nodes.forEach(node => {
        node.x = box.cx + (node.x - box.cx) * factor;
        node.y = box.cy + (node.y - box.cy) * factor;
        node.fx = node.x;
        node.fy = node.y;
      });
      renderPositions();
    }

    function boundingBox(nodes) {
      const xs = nodes.map(node => node.x).filter(Number.isFinite);
      const ys = nodes.map(node => node.y).filter(Number.isFinite);
      const minX = Math.min.apply(null, xs);
      const maxX = Math.max.apply(null, xs);
      const minY = Math.min.apply(null, ys);
      const maxY = Math.max.apply(null, ys);
      return {
        minX,
        maxX,
        minY,
        maxY,
        cx: (minX + maxX) / 2,
        cy: (minY + maxY) / 2
      };
    }

    function resetView() {
      svg.transition().duration(180).call(zoom.transform, d3.zoomIdentity);
    }

    function toggleLabels() {
      showLabels = !showLabels;
      applySizeScales();
      renderNetwork();
    }

    function updateSizes() {
      stopSimulation(false);
      applySizeScales();
      renderNetwork();
    }

    function applySizeScales() {
      const nodeScale = Number(document.getElementById("nodeScale").value);
      const labelScale = Number(document.getElementById("labelScale").value);
      currentNodes.forEach(node => {
        node.symbolSize = Math.max(2, Math.round((node._baseSymbolSize || 8) * nodeScale));
        const baseLabel = node.label || {};
        node.label = Object.assign({}, baseLabel, {
          show: showLabels,
          fontSize: Math.max(6, Math.round((node._baseFontSize || baseLabel.fontSize || 12) * labelScale))
        });
      });
    }

    function saveAsPng() {
      const clone = svg.node().cloneNode(true);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.setAttribute("width", String(width));
      clone.setAttribute("height", String(height));
      const background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      background.setAttribute("x", "0");
      background.setAttribute("y", "0");
      background.setAttribute("width", String(width));
      background.setAttribute("height", String(height));
      background.setAttribute("fill", "#ffffff");
      clone.insertBefore(background, clone.firstChild);
      const svgText = new XMLSerializer().serializeToString(clone);
      const image = new Image();
      const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = width * 3;
        canvas.height = height * 3;
        const context = canvas.getContext("2d");
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.scale(3, 3);
        context.drawImage(image, 0, 0);
        URL.revokeObjectURL(url);
        const link = document.createElement("a");
        link.download = "epb-hybrid-force-svg-network.png";
        link.href = canvas.toDataURL("image/png");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      };
      image.src = url;
    }
  </script>
</body>
</html>
"""
    html_path.write_text(html_text.replace("__DATA_FILE__", data_file), encoding="utf-8")


def find_author_community(temporal_results: dict[str, Any], author_name: str) -> int | None:
    partition = temporal_results.get("current_communities", {}).get("partition", {})
    if author_name in partition:
        return partition[author_name]
    author_lower = author_name.lower()
    for name, community_id in partition.items():
        if author_lower in name.lower() or name.lower() in author_lower:
            return community_id
    return None


def sanitize_html_file(path: Path):
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace('lang="zh"', 'lang="en"')
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"[\U0001F100-\U0001FAFF]\ufe0f?", "", text)
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)

    cleaned_lines = []
    for line in text.splitlines():
        if "//" in line:
            before, after = line.split("//", 1)
            if re.search(r"[\u4e00-\u9fff\U0001F100-\U0001FAFF]", after):
                line = before.rstrip()
        cleaned_lines.append(line)

    replacements = {
        " Reset Zoom": "Reset Zoom",
        " Save as PNG": "Save as PNG",
        " Toggle Labels": "Toggle Labels",
        " Show Legend": "Show Legend",
        " Hide Legend": "Hide Legend",
        " Lock Layout": "Lock Layout",
        " Unlock Layout": "Unlock Layout",
        " Author Generations": "Author Generations",
    }
    cleaned = "\n".join(cleaned_lines)
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    path.write_text(cleaned, encoding="utf-8")


def build_summary(
    input_csv: Path,
    builder: RevisedCSVNetworkBuilder,
    network: nx.Graph,
    lcc_network: nx.Graph,
    weighted_results: dict[str, Any],
    structural_results: dict[str, Any],
    small_world_results: dict[str, Any],
    communities_result: dict[str, Any],
    png_checks: list[dict[str, Any]],
    copied_figures: list[dict[str, str]],
    latest_figure_bundle: dict[str, Any],
) -> dict[str, Any]:
    raw = pd.read_csv(input_csv)
    review_counts = (
        raw.get("paper_needs_review", pd.Series(dtype=object))
        .fillna("<NA>")
        .value_counts()
        .to_dict()
    )

    full_components = [len(cc) for cc in nx.connected_components(network)]
    top_authors = sorted(
        (
            {
                "author": node,
                "paper_count": network.nodes[node].get("paper_count", 0),
                "weighted_degree": weighted_degree(network, node),
                "unweighted_degree": network.degree(node),
            }
            for node in network.nodes()
        ),
        key=lambda item: (
            item["weighted_degree"],
            item["paper_count"],
            item["unweighted_degree"],
            item["author"],
        ),
        reverse=True,
    )[:20]

    return {
        "run_date": "2026-06-09",
        "input_csv": str(input_csv),
        "input_sha256": file_sha256(input_csv),
        "paper_rows": len(raw),
        "year_min": int(pd.to_numeric(raw["year"], errors="coerce").min()),
        "year_max": int(pd.to_numeric(raw["year"], errors="coerce").max()),
        "paper_needs_review_counts": review_counts,
        "author_column": builder.author_col,
        "network_logic": {
            "edge_weight": "1 / number_of_unique_authors_on_paper",
            "node": "standardized author name from authors_full_final",
            "edge": "co-authorship pair within a paper",
            "lcc": "largest connected component of the author collaboration network",
            "path_analysis": "non-weighted shortest paths on the LCC",
            "community_detection": "Louvain on the LCC, weight='weight', random_seed=42",
        },
        "author_collaboration_network": {
            "nodes": network.number_of_nodes(),
            "edges": network.number_of_edges(),
            "density": nx.density(network),
            "connected_components": nx.number_connected_components(network),
            "largest_component_size": max(full_components) if full_components else 0,
        },
        "lcc_author_collaboration_network": {
            "nodes": lcc_network.number_of_nodes(),
            "edges": lcc_network.number_of_edges(),
            "density": nx.density(lcc_network),
            "share_of_author_collaboration_network_nodes": (
                lcc_network.number_of_nodes() / network.number_of_nodes()
                if network.number_of_nodes()
                else 0
            ),
        },
        "weighted_degree_stats": weighted_results["stats"],
        "weighted_degree_most_frequent": {
            f"{idx:.6f}": int(value)
            for idx, value in weighted_results["most_frequent"].head(10).items()
        },
        "structural_analysis": {
            network_name: {
                metric_name: {
                    k: v
                    for k, v in metric_data.items()
                    if k
                    not in {
                        "degrees",
                        "neighbor_degrees",
                        "clustering_by_degree",
                        "average_by_degree",
                        "group_sizes",
                    }
                }
                for metric_name, metric_data in network_data.items()
            }
            for network_name, network_data in structural_results.items()
        },
        "small_world": {
            "diameter": small_world_results.get("diameter"),
            "average_path_length": small_world_results.get("average_path_length"),
            "is_small_world_old_threshold": small_world_results.get("is_small_world"),
        },
        "communities": {
            "num_communities": communities_result.get("num_communities"),
            "modularity": communities_result.get("modularity"),
            "top_10_community_sizes": {
                str(cid): int(size)
                for cid, size in sorted(
                    communities_result.get("community_sizes", {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            },
        },
        "top_authors_by_weighted_degree": top_authors,
        "png_verification": png_checks,
        "copied_essay_figures": copied_figures,
        "latest_figure_bundle": latest_figure_bundle,
    }


def run(args: argparse.Namespace):
    random.seed(42)
    np.random.seed(42)

    input_csv = Path(args.input_csv).resolve()
    run_dir = Path(args.run_dir).resolve()
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    metrics_dir = run_dir / "metrics"
    for directory in [figures_dir, tables_dir, metrics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EPB revised author collaboration network analysis")
    print("=" * 72)
    print(f"Input CSV: {input_csv}")
    print(f"Run dir:   {run_dir}")

    builder = RevisedCSVNetworkBuilder(input_csv, author_col=args.author_col)
    network = builder.build_collaboration_network(use_weighted=True)
    lcc_network = largest_connected_component_subgraph(network)
    print(
        "LCC of author collaboration network: "
        f"{lcc_network.number_of_nodes()} nodes, {lcc_network.number_of_edges()} edges"
    )

    full_analyzer = NetworkAnalyzer(network)
    lcc_analyzer = NetworkAnalyzer(lcc_network)

    print("\nGenerating weighted-degree analysis.")
    np.random.seed(42)
    weighted_results = run_silent(
        full_analyzer.analyze_weighted_degree_distribution_detailed,
        plot=True, save_path=str(figures_dir / "03-weighted-degree-analysis.png")
    )
    plt.close("all")

    print("\nGenerating weighted/unweighted comparison.")
    run_silent(
        full_analyzer.plot_weighted_comparison,
        top_n=15, save_path=str(figures_dir / "04-weighted-comparison.png")
    )
    plt.close("all")

    print("\nGenerating structural analysis.")
    structural_results = create_author_network_structural_plot(
        network,
        lcc_network,
        figures_dir / "08-structural-analysis.png",
    )
    plt.close("all")

    print("\nGenerating small-world analysis.")
    structural_lcc = run_silent(NetworkStructuralAnalyzer, lcc_network)
    small_world_results = render_small_world_clean(
        lcc_network,
        figures_dir / "14-small-world-largest-component.png",
    )
    plt.close("all")

    print("\nDetecting communities.")
    communities_result = run_silent(
        structural_lcc.detect_communities,
        method="louvain", plot=False, random_seed=42, standardize_ids=True
    )
    if not communities_result:
        raise RuntimeError("Community detection failed")
    run_silent(
        structural_lcc.save_community_results,
        communities_result, str(figures_dir / "community-detection-results.json")
    )

    print("\nGenerating ECharts network visualization.")
    # The inherited ECharts helper expects an attribute named filtered_network.
    # For the revised manuscript workflow this compatibility slot holds the LCC
    # of the author collaboration network.
    lcc_analyzer.filtered_network = lcc_network
    network_data_file, network_html = run_silent(
        lcc_analyzer.create_echarts_visualization,
        communities_result=communities_result,
        max_nodes=args.echarts_max_nodes,
        output_dir=str(figures_dir),
        node_size_scale=10.0,
        node_size_min=18,
        node_size_max=160,
        label_font_min=12,
        label_font_max=24,
    )
    write_manual_echarts_network_html(Path(network_html), Path(network_data_file).name)
    write_hybrid_force_svg_network_html(
        figures_dir / "01-network-hybrid-force-pack.html",
        Path(network_data_file).name,
    )

    network_png = figures_dir / "01-epbnetwork.png"
    ok = False
    if not args.disable_echarts_auto_render:
        ok = render_echarts_force_png(
            lcc_network,
            communities_result,
            network_png,
            figures_dir,
            max_components=args.echarts_render_components,
            max_nodes=args.echarts_render_max_nodes,
            label_count=args.echarts_render_labels,
            settle_ms=args.echarts_render_settle_ms,
            viewport=(args.echarts_render_width, args.echarts_render_height),
            pixel_ratio=args.echarts_render_pixel_ratio,
        )
    if not ok and args.use_browser_screenshots:
        ok = screenshot_html_element(
            Path(network_html),
            "#network-chart",
            network_png,
            viewport=(2200, 1400),
            wait_ms=args.html_wait_ms,
        )
    if not ok:
        render_static_network_png(
            lcc_network, communities_result, network_png, max_nodes=args.static_max_nodes
        )

    sanitize_html_file(Path(network_html))

    optional_pngs = []
    if not args.disable_cosmograph_candidate:
        print("\nGenerating Cosmograph candidate visualization.")
        ok = render_cosmograph_candidate_png(
            lcc_network,
            communities_result,
            figures_dir,
            max_components=args.cosmograph_render_components,
            max_nodes=args.cosmograph_render_max_nodes,
            label_count=args.cosmograph_render_labels,
            viewport=(args.cosmograph_render_width, args.cosmograph_render_height),
            wait_ms=args.cosmograph_render_wait_ms,
        )
        if ok:
            optional_pngs.append(figures_dir / "01-epbnetwork_cosmograph_candidate.png")

    print("\nGenerating community temporal evolution.")
    target_communities = top_community_ids(communities_result, n=10)
    temporal_results = run_silent(
        structural_lcc.analyze_community_temporal_evolution,
        target_communities=target_communities,
        top_n_leaders=5,
        decades=["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"],
        save_figures=True,
        figures_dir=str(figures_dir),
    )
    if temporal_results is None:
        raise RuntimeError("Community temporal evolution failed")
    plt.close("all")
    render_community_decade_evolution_clean(
        lcc_network,
        temporal_results,
        figures_dir / "02-community_decade_evolution.png",
    )

    community_html = figures_dir / "06-community-evolution-network.html"
    community_png = figures_dir / "05-community_evolution_network.png"
    batty_community = find_author_community(temporal_results, "Batty, Michael")
    if batty_community is None:
        raise RuntimeError("Could not locate Michael Batty's community for the generation network")

    generation_data = build_generation_network_data(
        structural_lcc,
        temporal_results,
        batty_community,
        ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"],
    )
    write_generation_network_svg_html(community_html, generation_data)
    ok = screenshot_html_element(
        community_html,
        "#export-frame",
        community_png,
        viewport=(2400, 1400),
        wait_ms=500,
    )
    if not ok:
        raise RuntimeError("Browser SVG screenshot failed for the community generation network")

    sanitize_html_file(community_html)

    cleanup_obsolete_html(figures_dir)

    print("\nExporting supporting tables.")
    export_top_authors(network, tables_dir / "top_authors_by_weighted_degree.csv")
    export_community_representatives(
        lcc_network,
        communities_result,
        tables_dir / "community_representatives.csv",
    )

    expected_pngs = [
        figures_dir / "01-epbnetwork.png",
        figures_dir / "02-community_decade_evolution.png",
        figures_dir / "03-weighted-degree-analysis.png",
        figures_dir / "04-weighted-comparison.png",
        figures_dir / "05-community_evolution_network.png",
        figures_dir / "08-structural-analysis.png",
        figures_dir / "14-small-world-largest-component.png",
    ]
    png_checks = [verify_png(path) for path in expected_pngs + optional_pngs]
    for check in png_checks:
        if check["width"] < 500 or check["height"] < 300 or check["nonwhite_ratio"] < 0.01:
            raise RuntimeError(f"PNG verification failed: {check}")

    copied_figures = []
    if args.update_essay:
        copied_figures = copy_essay_figures(figures_dir, Path(args.essay_figures_dir).resolve())

    latest_figure_bundle = {}
    if not args.skip_latest_figures_bundle:
        latest_figure_bundle = copy_latest_figure_bundle(
            figures_dir,
            Path(args.latest_figures_dir).resolve(),
            run_dir,
        )

    summary = build_summary(
        input_csv=input_csv,
        builder=builder,
        network=network,
        lcc_network=lcc_network,
        weighted_results=weighted_results,
        structural_results=structural_results,
        small_world_results=small_world_results,
        communities_result=communities_result,
        png_checks=png_checks,
        copied_figures=copied_figures,
        latest_figure_bundle=latest_figure_bundle,
    )
    summary_path = metrics_dir / "network_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print("Revised network analysis complete.")


def render_generation_network_fallback(data: dict[str, Any], output_path: Path):
    print(f"Rendering generation network PNG: {output_path}")
    graph = nx.Graph()
    for node in data.get("nodes", []):
        graph.add_node(node["id"], **node)
    for link in data.get("links", []):
        graph.add_edge(link["source"], link["target"], weight=link.get("weight", 1))
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot render empty generation network")

    pos = nx.spring_layout(graph, seed=42, weight="weight", iterations=150, k=0.55)
    fig, ax = plt.subplots(figsize=(20, 12), facecolor="white")
    edge_widths = [0.8 + min(graph[u][v].get("weight", 1), 2.5) for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, edge_color="#7f7f7f", alpha=0.55)
    node_colors = [graph.nodes[n].get("color", "#95a5a6") for n in graph.nodes()]
    node_sizes = [max(35, graph.nodes[n].get("symbolSize", 12) * 10) for n in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=1.0,
    )
    labels = {node: graph.nodes[node].get("name", display_surname(node)) for node in graph.nodes()}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        ax=ax,
        font_size=9,
        font_color="#2c3e50",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#6aa58d", "alpha": 0.8},
    )

    decade_colors = data.get("decade_colors", {})
    handles = []
    for decade in ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s", "unknown"]:
        count = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("first_decade") == decade)
        if count:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=decade_colors.get(decade, "#95a5a6"),
                    markersize=11,
                    label=f"{decade} ({count})",
                )
            )
    if handles:
        ax.legend(handles=handles, title="Author Generations", loc="lower right", frameon=True)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_generation_network_data(
    structural_analyzer: NetworkStructuralAnalyzer,
    temporal_results: dict[str, Any],
    community_id: int,
    decades: list[str],
) -> dict[str, Any]:
    """Build corrected community-generation data for a static PNG.

    The original HTML helper initializes every community member as "unknown"
    before checking active decades, which can leave generation counts at zero.
    This builder derives the first active decade directly from node paper data.
    """
    graph = structural_analyzer.network
    partition = temporal_results.get("current_communities", {}).get("partition", {})
    members = [node for node, cid in partition.items() if cid == community_id and graph.has_node(node)]
    decade_colors = {
        "1970s": "#5470c6",
        "1980s": "#91cc75",
        "1990s": "#fac858",
        "2000s": "#ee6666",
        "2010s": "#73c0de",
        "2020s": "#3ba272",
        "unknown": "#95a5a6",
    }

    nodes = []
    first_decade_by_author = {}
    for member in members:
        papers = graph.nodes[member].get("papers", [])
        paper_decades = {paper.get("decade") for paper in papers}
        first_decade = next((decade for decade in decades if decade in paper_decades), "unknown")
        first_decade_by_author[member] = first_decade
        degree = graph.degree(member)
        nodes.append(
            {
                "id": member,
                "name": display_surname(member),
                "full_name": member,
                "first_decade": first_decade,
                "color": decade_colors.get(first_decade, decade_colors["unknown"]),
                "papers": graph.nodes[member].get("paper_count", 0),
                "degree": degree,
                "symbolSize": min(max(degree * 4, 12), 36),
            }
        )

    links = []
    member_set = set(members)
    for source, target, attrs in graph.edges(members, data=True):
        if source not in member_set or target not in member_set:
            continue
        source_decade = first_decade_by_author.get(source, "unknown")
        target_decade = first_decade_by_author.get(target, "unknown")
        edge_type = "intra-decade" if source_decade == target_decade else "inter-decade"
        links.append(
            {
                "source": source,
                "target": target,
                "weight": attrs.get("weight", 1),
                "type": edge_type,
                "lineStyle": {
                    "color": "#808080",
                    "width": min(max(attrs.get("weight", 1) * 2, 2), 6),
                    "opacity": 0.72 if edge_type == "intra-decade" else 0.9,
                },
            }
        )

    return {
        "nodes": nodes,
        "links": links,
        "community_id": community_id,
        "decades": decades,
        "decade_colors": decade_colors,
        "stats": {
            "total_authors": len(nodes),
            "total_collaborations": len(links),
            "inter_decade_collaborations": sum(1 for link in links if link["type"] == "inter-decade"),
            "intra_decade_collaborations": sum(1 for link in links if link["type"] == "intra-decade"),
        },
    }


def write_generation_network_svg_html(html_path: Path, data: dict[str, Any]) -> None:
    """Write a browser-rendered SVG version of the community generation network."""
    width = 2400
    height = 1400
    margin = 90
    legend_width = 310
    content_width = width - legend_width - margin * 2
    content_height = height - margin * 2

    graph = nx.Graph()
    for node in data.get("nodes", []):
        graph.add_node(node["id"], **node)
    for link in data.get("links", []):
        graph.add_edge(link["source"], link["target"], **link)

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot render an empty generation network")

    positions = nx.spring_layout(
        graph,
        seed=42,
        weight="weight",
        k=1.85 / max(np.sqrt(graph.number_of_nodes()), 1),
        iterations=900,
    )
    xs = [coord[0] for coord in positions.values()]
    ys = [coord[1] for coord in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def scaled_position(node_id: str) -> tuple[float, float]:
        x, y = positions[node_id]
        sx = margin + ((x - min_x) / span_x) * content_width
        sy = margin + ((y - min_y) / span_y) * content_height
        return sx, sy

    decade_colors = data.get("decade_colors", {})
    decades = [decade for decade in data.get("decades", []) if decade in decade_colors]
    generation_counts = {
        decade: sum(1 for node in data.get("nodes", []) if node.get("first_decade") == decade)
        for decade in decades
    }
    if any(node.get("first_decade") == "unknown" for node in data.get("nodes", [])):
        decades = [*decades, "unknown"]
        generation_counts["unknown"] = sum(
            1 for node in data.get("nodes", []) if node.get("first_decade") == "unknown"
        )

    node_by_id = {node["id"]: node for node in data.get("nodes", [])}
    center_x = margin + content_width / 2
    center_y = margin + content_height / 2

    link_parts = []
    for link in data.get("links", []):
        if link["source"] not in positions or link["target"] not in positions:
            continue
        x1, y1 = scaled_position(link["source"])
        x2, y2 = scaled_position(link["target"])
        source_decade = node_by_id.get(link["source"], {}).get("first_decade")
        target_decade = node_by_id.get(link["target"], {}).get("first_decade")
        is_cross = source_decade != target_decade
        color = "#7c3aed" if is_cross else "#8d98a3"
        opacity = 0.38 if is_cross else 0.28
        width_px = min(max(float(link.get("weight", 1.0)) * 2.8, 1.2), 5.5)
        link_parts.append(
            f'<line class="link" data-source="{html_escape(str(link["source"]))}" '
            f'data-target="{html_escape(str(link["target"]))}" '
            f'x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width_px:.2f}" stroke-opacity="{opacity:.2f}" />'
        )

    node_parts = []
    for node in sorted(data.get("nodes", []), key=lambda item: item["id"]):
        node_id = node["id"]
        if node_id not in positions:
            continue
        x, y = scaled_position(node_id)
        radius = min(max(float(node.get("symbolSize", 14)) * 0.42, 5.5), 16)
        color = node.get("color") or decade_colors.get(node.get("first_decade"), "#64748b")
        is_key = "Batty" in node_id or node.get("degree", 0) >= 10
        font_size = 17 if not is_key else 21
        label = html_escape(node.get("name", node_id))
        full_name = html_escape(node.get("full_name", node_id))
        label_width = max(34, len(label) * font_size * 0.58 + 14)
        label_height = font_size + 9
        anchor_right = x <= center_x
        label_gap = radius + 8
        label_x = label_gap if anchor_right else -label_gap - label_width
        label_y = -label_height / 2
        safe_id = html_escape(str(node_id))
        safe_decade = html_escape(str(node.get("first_decade", "unknown")))
        safe_papers = html_escape(str(node.get("papers", "")))
        safe_degree = html_escape(str(node.get("degree", "")))

        node_parts.append(
            f'<g class="node" transform="translate({x:.2f} {y:.2f})" '
            f'data-id="{safe_id}" data-full-name="{full_name}" data-decade="{safe_decade}" '
            f'data-papers="{safe_papers}" data-degree="{safe_degree}" '
            f'data-x="{x:.2f}" data-y="{y:.2f}">'
            f'<circle cx="0" cy="0" r="{radius:.2f}" fill="{color}" '
            f'stroke="#ffffff" stroke-width="2.2" />'
            f'<g class="node-label">'
            f'<rect x="{label_x:.2f}" y="{label_y:.2f}" width="{label_width:.2f}" '
            f'height="{label_height:.2f}" rx="5" ry="5" />'
            f'<text x="{label_x + label_width / 2:.2f}" y="{label_y + font_size:.2f}" '
            f'font-size="{font_size}" font-weight="{"700" if is_key else "600"}">{label}</text>'
            f'</g></g>'
        )

    legend_x = width - legend_width + 35
    legend_y = 105
    legend_items = []
    for idx, decade in enumerate(decades):
        y = legend_y + 58 + idx * 42
        color = decade_colors.get(decade, "#95a5a6")
        count = generation_counts.get(decade, 0)
        legend_items.append(
            f'<circle cx="{legend_x}" cy="{y}" r="10" fill="{color}" />'
            f'<text x="{legend_x + 24}" y="{y + 6}" class="legend-text">{html_escape(decade)} ({count})</text>'
        )

    title = f"Community {data.get('community_id', '')} Author Generation Network"
    stats = data.get("stats", {})
    subtitle = (
        f"{stats.get('total_authors', 0)} authors, "
        f"{stats.get('total_collaborations', 0)} collaborations, "
        f"{stats.get('inter_decade_collaborations', 0)} cross-decade links"
    )
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_escape(title)}</title>
  <style>
    html, body {{
      margin: 0;
      background: #ffffff;
      color: #1f2937;
      font-family: Arial, Helvetica, sans-serif;
    }}
    #export-frame {{
      width: 100vw;
      height: 100vh;
      min-width: 960px;
      min-height: 560px;
      background: #ffffff;
      overflow: hidden;
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
      background: #ffffff;
      cursor: grab;
      user-select: none;
    }}
    svg.is-panning {{
      cursor: grabbing;
    }}
    .title {{
      font-size: 30px;
      font-weight: 700;
      fill: #111827;
    }}
    .subtitle {{
      font-size: 17px;
      font-weight: 500;
      fill: #64748b;
    }}
    .node-label rect {{
      fill: rgba(255,255,255,0.94);
      stroke: #9db8b2;
      stroke-width: 1.6;
    }}
    .node-label text {{
      fill: #263241;
      text-anchor: middle;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 2px;
      stroke-linejoin: round;
    }}
    .node {{
      cursor: pointer;
      transition: opacity 0.16s ease;
    }}
    .node circle {{
      transition: stroke-width 0.16s ease, filter 0.16s ease, opacity 0.16s ease;
    }}
    .node.active circle,
    .node.locked circle {{
      stroke: #111827;
      stroke-width: 3.5;
      filter: drop-shadow(0 5px 8px rgba(15, 23, 42, 0.25));
    }}
    .node.selected circle {{
      stroke: #f59e0b;
      stroke-width: 4.2;
      filter: drop-shadow(0 5px 9px rgba(245, 158, 11, 0.32));
    }}
    .node.dimmed,
    .link.dimmed {{
      opacity: 0.12;
    }}
    .link {{
      transition: opacity 0.16s ease, stroke-width 0.16s ease;
    }}
    .link.active {{
      stroke-opacity: 0.86;
      stroke-width: 4.5;
    }}
    .legend-title {{
      font-size: 19px;
      font-weight: 700;
      fill: #111827;
    }}
    .legend-text {{
      font-size: 17px;
      font-weight: 600;
      fill: #334155;
    }}
    .legend {{
      cursor: move;
      user-select: none;
    }}
    .legend.dragging {{
      filter: drop-shadow(0 10px 18px rgba(15, 23, 42, 0.18));
    }}
    .selection-rect {{
      fill: rgba(37, 99, 235, 0.10);
      stroke: #2563eb;
      stroke-width: 2;
      stroke-dasharray: 8 6;
      pointer-events: none;
    }}
    #tooltip {{
      position: fixed;
      z-index: 20;
      display: none;
      min-width: 220px;
      max-width: 320px;
      padding: 12px 14px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 14px 30px rgba(15, 23, 42, 0.16);
      color: #1f2937;
      font-size: 14px;
      line-height: 1.45;
      pointer-events: none;
    }}
    #tooltip strong {{
      display: block;
      margin-bottom: 7px;
      font-size: 16px;
      color: #111827;
    }}
  </style>
</head>
<body>
  <div id="export-frame">
    <svg id="generation-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html_escape(title)}">
      <rect class="background" x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
      <text x="{margin}" y="48" class="title">{html_escape(title)}</text>
      <text x="{margin}" y="78" class="subtitle">{html_escape(subtitle)}</text>
      <g id="graph-layer">
        <g class="links">
          {"".join(link_parts)}
        </g>
        <g class="nodes">
          {"".join(node_parts)}
        </g>
      </g>
      <rect id="selection-rect" class="selection-rect" x="0" y="0" width="0" height="0" visibility="hidden" />
      <g class="legend" data-x="0" data-y="0">
        <rect x="{width - legend_width + 14}" y="72" width="{legend_width - 42}" height="{max(175, 86 + len(decades) * 42)}" rx="8" fill="#ffffff" stroke="#d7dee8" />
        <text x="{legend_x - 12}" y="{legend_y}" class="legend-title">Author Generations</text>
        {"".join(legend_items)}
      </g>
    </svg>
  </div>
  <div id="tooltip"></div>
  <script>
    const svg = document.getElementById("generation-svg");
    const tooltip = document.getElementById("tooltip");
    const initialViewBox = {{ x: 0, y: 0, width: {width}, height: {height} }};
    let viewBox = {{ ...initialViewBox }};
    let lockedNodeId = null;
    let dragState = null;
    let panStart = null;
    let panMoved = false;
    let selectionState = null;
    let legendDragState = null;
    let selectionJustCompleted = false;
    let suppressNextClick = false;
    let suppressBackgroundClick = false;
    const selectedNodeIds = new Set();
    const selectionRect = document.getElementById("selection-rect");
    const legend = document.querySelector(".legend");
    const STORAGE_KEY = "epb-community-generation-network:community-{data.get('community_id', '')}:nodes-{len(data.get('nodes', []))}:links-{len(data.get('links', []))}:v1";

    const nodes = Array.from(document.querySelectorAll(".node"));
    const links = Array.from(document.querySelectorAll(".link"));
    const nodeById = new Map(nodes.map(node => [node.dataset.id, node]));
    const initialNodePositions = new Map(
      nodes.map(node => [node.dataset.id, {{ x: Number(node.dataset.x), y: Number(node.dataset.y) }}])
    );
    const neighborMap = new Map(nodes.map(node => [node.dataset.id, new Set([node.dataset.id])]));

    links.forEach(link => {{
      const source = link.dataset.source;
      const target = link.dataset.target;
      if (neighborMap.has(source)) neighborMap.get(source).add(target);
      if (neighborMap.has(target)) neighborMap.get(target).add(source);
    }});

    function setViewBox() {{
      svg.setAttribute("viewBox", `${{viewBox.x}} ${{viewBox.y}} ${{viewBox.width}} ${{viewBox.height}}`);
    }}

    function svgPoint(event) {{
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      return point.matrixTransform(svg.getScreenCTM().inverse());
    }}

    function setHighlight(nodeId) {{
      const neighbors = neighborMap.get(nodeId) || new Set([nodeId]);
      nodes.forEach(node => {{
        const active = neighbors.has(node.dataset.id);
        node.classList.toggle("dimmed", !active);
        node.classList.toggle("active", active);
        node.classList.toggle("locked", node.dataset.id === lockedNodeId);
      }});
      links.forEach(link => {{
        const active = link.dataset.source === nodeId || link.dataset.target === nodeId;
        link.classList.toggle("dimmed", !active);
        link.classList.toggle("active", active);
      }});
      updateSelectionStyles();
    }}

    function updateSelectionStyles() {{
      nodes.forEach(node => {{
        node.classList.toggle("selected", selectedNodeIds.has(node.dataset.id));
      }});
    }}

    function clearHighlight() {{
      if (lockedNodeId) {{
        setHighlight(lockedNodeId);
        return;
      }}
      nodes.forEach(node => node.classList.remove("dimmed", "active", "locked"));
      links.forEach(link => link.classList.remove("dimmed", "active"));
      updateSelectionStyles();
    }}

    function clearSelection() {{
      selectedNodeIds.clear();
      updateSelectionStyles();
    }}

    function showTooltip(node, event) {{
      tooltip.innerHTML = `
        <strong>${{node.dataset.fullName}}</strong>
        First active decade: ${{node.dataset.decade}}<br>
        Papers: ${{node.dataset.papers}}<br>
        Collaborators: ${{node.dataset.degree}}
      `;
      tooltip.style.display = "block";
      moveTooltip(event);
    }}

    function moveTooltip(event) {{
      const pad = 18;
      const x = Math.min(event.clientX + pad, window.innerWidth - tooltip.offsetWidth - pad);
      const y = Math.min(event.clientY + pad, window.innerHeight - tooltip.offsetHeight - pad);
      tooltip.style.left = `${{Math.max(pad, x)}}px`;
      tooltip.style.top = `${{Math.max(pad, y)}}px`;
    }}

    function hideTooltip() {{
      if (!lockedNodeId) tooltip.style.display = "none";
    }}

    function updateNodePosition(node, x, y) {{
      node.dataset.x = String(x);
      node.dataset.y = String(y);
      node.setAttribute("transform", `translate(${{x}} ${{y}})`);
      links.forEach(link => {{
        if (link.dataset.source === node.dataset.id) {{
          link.setAttribute("x1", String(x));
          link.setAttribute("y1", String(y));
        }}
        if (link.dataset.target === node.dataset.id) {{
          link.setAttribute("x2", String(x));
          link.setAttribute("y2", String(y));
        }}
      }});
    }}

    function updateLegendPosition(x, y) {{
      legend.dataset.x = String(x);
      legend.dataset.y = String(y);
      legend.setAttribute("transform", `translate(${{x}} ${{y}})`);
    }}

    function saveState() {{
      try {{
        const nodePositions = {{}};
        nodes.forEach(node => {{
          nodePositions[node.dataset.id] = {{
            x: Number(node.dataset.x),
            y: Number(node.dataset.y),
          }};
        }});
        localStorage.setItem(STORAGE_KEY, JSON.stringify({{
          version: 1,
          viewBox,
          legend: {{
            x: Number(legend.dataset.x || 0),
            y: Number(legend.dataset.y || 0),
          }},
          nodePositions,
        }}));
      }} catch (error) {{
        console.warn("Could not save manual layout state", error);
      }}
    }}

    function restoreState() {{
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const state = JSON.parse(raw);
        if (state.nodePositions) {{
          Object.entries(state.nodePositions).forEach(([id, pos]) => {{
            const node = nodeById.get(id);
            if (node && Number.isFinite(pos.x) && Number.isFinite(pos.y)) {{
              updateNodePosition(node, pos.x, pos.y);
            }}
          }});
        }}
        if (
          state.legend &&
          Number.isFinite(state.legend.x) &&
          Number.isFinite(state.legend.y)
        ) {{
          updateLegendPosition(state.legend.x, state.legend.y);
        }}
        if (
          state.viewBox &&
          Number.isFinite(state.viewBox.x) &&
          Number.isFinite(state.viewBox.y) &&
          Number.isFinite(state.viewBox.width) &&
          Number.isFinite(state.viewBox.height)
        ) {{
          viewBox = {{ ...state.viewBox }};
          setViewBox();
        }}
      }} catch (error) {{
        console.warn("Could not restore manual layout state", error);
      }}
    }}

    function resetManualState() {{
      try {{
        localStorage.removeItem(STORAGE_KEY);
      }} catch (error) {{
        console.warn("Could not clear manual layout state", error);
      }}
      initialNodePositions.forEach((pos, id) => {{
        const node = nodeById.get(id);
        if (node) updateNodePosition(node, pos.x, pos.y);
      }});
      updateLegendPosition(0, 0);
      viewBox = {{ ...initialViewBox }};
      lockedNodeId = null;
      tooltip.style.display = "none";
      setViewBox();
      clearSelection();
      clearHighlight();
    }}

    restoreState();

    legend.addEventListener("pointerdown", event => {{
      if (event.button !== 0) return;
      event.stopPropagation();
      const point = svgPoint(event);
      legendDragState = {{
        start: point,
        x: Number(legend.dataset.x || 0),
        y: Number(legend.dataset.y || 0),
      }};
      legend.classList.add("dragging");
      legend.setPointerCapture(event.pointerId);
    }});

    nodes.forEach(node => {{
      node.addEventListener("mouseenter", event => {{
        if (!lockedNodeId) setHighlight(node.dataset.id);
        showTooltip(node, event);
      }});
      node.addEventListener("mousemove", event => moveTooltip(event));
      node.addEventListener("mouseleave", () => {{
        hideTooltip();
        if (!lockedNodeId) clearHighlight();
      }});
      node.addEventListener("click", event => {{
        event.stopPropagation();
        if (suppressNextClick) {{
          suppressNextClick = false;
          return;
        }}
        if (event.shiftKey) {{
          lockedNodeId = null;
          tooltip.style.display = "none";
          if (selectedNodeIds.has(node.dataset.id)) {{
            selectedNodeIds.delete(node.dataset.id);
          }} else {{
            selectedNodeIds.add(node.dataset.id);
          }}
          clearHighlight();
          return;
        }}
        lockedNodeId = lockedNodeId === node.dataset.id ? null : node.dataset.id;
        if (lockedNodeId) {{
          setHighlight(lockedNodeId);
        }} else {{
          tooltip.style.display = "none";
          clearHighlight();
        }}
      }});
      node.addEventListener("pointerdown", event => {{
        if (event.button !== 0) return;
        event.stopPropagation();
        const point = svgPoint(event);
        const nodeId = node.dataset.id;
        const dragIds = selectedNodeIds.has(nodeId) ? Array.from(selectedNodeIds) : [nodeId];
        dragState = {{
          ids: dragIds,
          start: point,
          moved: false,
          startPositions: new Map(
            dragIds.map(id => {{
              const target = nodeById.get(id);
              return [id, {{ x: Number(target.dataset.x), y: Number(target.dataset.y) }}];
            }})
          ),
        }};
        node.setPointerCapture(event.pointerId);
      }});
    }});

    svg.addEventListener("pointerdown", event => {{
      if (event.target.closest && event.target.closest(".node")) return;
      const point = svgPoint(event);
      if (event.shiftKey) {{
        selectionState = {{
          start: point,
          additive: true,
        }};
        selectionRect.setAttribute("x", String(point.x));
        selectionRect.setAttribute("y", String(point.y));
        selectionRect.setAttribute("width", "0");
        selectionRect.setAttribute("height", "0");
        selectionRect.setAttribute("visibility", "visible");
        return;
      }}
      panStart = {{
        clientX: event.clientX,
        clientY: event.clientY,
        viewBox: {{ ...viewBox }},
      }};
      panMoved = false;
      svg.classList.add("is-panning");
    }});

    svg.addEventListener("pointermove", event => {{
      if (legendDragState) {{
        const point = svgPoint(event);
        updateLegendPosition(
          legendDragState.x + point.x - legendDragState.start.x,
          legendDragState.y + point.y - legendDragState.start.y,
        );
        event.preventDefault();
        return;
      }}
      if (selectionState) {{
        const point = svgPoint(event);
        const x = Math.min(selectionState.start.x, point.x);
        const y = Math.min(selectionState.start.y, point.y);
        const rectWidth = Math.abs(point.x - selectionState.start.x);
        const rectHeight = Math.abs(point.y - selectionState.start.y);
        selectionRect.setAttribute("x", String(x));
        selectionRect.setAttribute("y", String(y));
        selectionRect.setAttribute("width", String(rectWidth));
        selectionRect.setAttribute("height", String(rectHeight));
        event.preventDefault();
        return;
      }}
      if (dragState) {{
        const point = svgPoint(event);
        const dx = point.x - dragState.start.x;
        const dy = point.y - dragState.start.y;
        if (Math.hypot(dx, dy) > 2) dragState.moved = true;
        dragState.ids.forEach(id => {{
          const target = nodeById.get(id);
          const start = dragState.startPositions.get(id);
          updateNodePosition(target, start.x + dx, start.y + dy);
        }});
        event.preventDefault();
        return;
      }}
      if (panStart) {{
        const dx = event.clientX - panStart.clientX;
        const dy = event.clientY - panStart.clientY;
        if (Math.hypot(dx, dy) > 3) panMoved = true;
        const rect = svg.getBoundingClientRect();
        viewBox.x = panStart.viewBox.x - dx * panStart.viewBox.width / rect.width;
        viewBox.y = panStart.viewBox.y - dy * panStart.viewBox.height / rect.height;
        setViewBox();
      }}
    }});

    svg.addEventListener("pointerup", () => {{
      if (legendDragState) {{
        legendDragState = null;
        legend.classList.remove("dragging");
        saveState();
      }}
      if (selectionState) {{
        const x = Number(selectionRect.getAttribute("x"));
        const y = Number(selectionRect.getAttribute("y"));
        const rectWidth = Number(selectionRect.getAttribute("width"));
        const rectHeight = Number(selectionRect.getAttribute("height"));
        const x2 = x + rectWidth;
        const y2 = y + rectHeight;
        nodes.forEach(node => {{
          const nodeX = Number(node.dataset.x);
          const nodeY = Number(node.dataset.y);
          if (nodeX >= x && nodeX <= x2 && nodeY >= y && nodeY <= y2) {{
            selectedNodeIds.add(node.dataset.id);
          }}
        }});
        selectionRect.setAttribute("visibility", "hidden");
        selectionState = null;
        selectionJustCompleted = true;
        lockedNodeId = null;
        tooltip.style.display = "none";
        clearHighlight();
        setTimeout(() => {{ selectionJustCompleted = false; }}, 0);
      }}
      if (dragState) {{
        if (dragState.moved) suppressNextClick = true;
        if (dragState.moved) saveState();
        dragState = null;
      }}
      if (panStart && panMoved) suppressBackgroundClick = true;
      if (panStart && panMoved) saveState();
      panStart = null;
      panMoved = false;
      svg.classList.remove("is-panning");
    }});
    svg.addEventListener("pointerleave", () => {{
      if (legendDragState || (dragState && dragState.moved) || (panStart && panMoved)) {{
        saveState();
      }}
      legendDragState = null;
      legend.classList.remove("dragging");
      dragState = null;
      if (selectionState) {{
        selectionRect.setAttribute("visibility", "hidden");
        selectionState = null;
      }}
      panStart = null;
      panMoved = false;
      svg.classList.remove("is-panning");
    }});

    svg.addEventListener("wheel", event => {{
      event.preventDefault();
      const point = svgPoint(event);
      const factor = event.deltaY > 0 ? 1.12 : 0.88;
      const nextWidth = Math.min(Math.max(viewBox.width * factor, initialViewBox.width * 0.25), initialViewBox.width * 3.2);
      const nextHeight = nextWidth * initialViewBox.height / initialViewBox.width;
      const ratio = nextWidth / viewBox.width;
      viewBox.x = point.x - (point.x - viewBox.x) * ratio;
      viewBox.y = point.y - (point.y - viewBox.y) * ratio;
      viewBox.width = nextWidth;
      viewBox.height = nextHeight;
      setViewBox();
      saveState();
    }}, {{ passive: false }});

    svg.addEventListener("dblclick", event => {{
      if (event.shiftKey) {{
        resetManualState();
        return;
      }}
      viewBox = {{ ...initialViewBox }};
      lockedNodeId = null;
      tooltip.style.display = "none";
      setViewBox();
      clearSelection();
      clearHighlight();
      saveState();
    }});

    svg.addEventListener("click", event => {{
      if (event.target.closest && event.target.closest(".node")) return;
      if (selectionJustCompleted || suppressBackgroundClick) {{
        selectionJustCompleted = false;
        suppressBackgroundClick = false;
        return;
      }}
      if (!event.shiftKey) {{
        lockedNodeId = null;
        tooltip.style.display = "none";
        clearSelection();
        clearHighlight();
      }}
    }});

    document.addEventListener("keydown", event => {{
      if (event.key === "Escape") {{
        lockedNodeId = null;
        tooltip.style.display = "none";
        clearSelection();
        clearHighlight();
      }}
    }});

    window.__renderReady = true;
  </script>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run EPB author collaboration network analysis on the revised CSV."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--author-col", default="authors_full_final")
    parser.add_argument("--essay-figures-dir", default=str(DEFAULT_ESSAY_FIGURES))
    parser.add_argument("--latest-figures-dir", default=str(DEFAULT_LATEST_FIGURES))
    parser.add_argument("--update-essay", action="store_true")
    parser.add_argument(
        "--skip-latest-figures-bundle",
        action="store_true",
        help="Skip copying generated figure files into the centralized latest figure bundle.",
    )
    parser.add_argument("--echarts-max-nodes", type=int, default=1500)
    parser.add_argument("--static-max-nodes", type=int, default=1600)
    parser.add_argument("--html-wait-ms", type=int, default=14000)
    parser.add_argument("--echarts-render-components", type=int, default=1)
    parser.add_argument("--echarts-render-max-nodes", type=int, default=500)
    parser.add_argument(
        "--echarts-render-labels",
        type=int,
        default=85,
        help="Number of labels to show in the ECharts PNG. Use <=0 for all labels.",
    )
    parser.add_argument("--echarts-render-settle-ms", type=int, default=18000)
    parser.add_argument("--echarts-render-width", type=int, default=3200)
    parser.add_argument("--echarts-render-height", type=int, default=1900)
    parser.add_argument("--echarts-render-pixel-ratio", type=int, default=2)
    parser.add_argument(
        "--disable-echarts-auto-render",
        action="store_true",
        help="Disable automated ECharts PNG rendering and use the static fallback path.",
    )
    parser.add_argument(
        "--disable-cosmograph-candidate",
        action="store_true",
        help="Skip the separate Cosmograph candidate render.",
    )
    parser.add_argument("--cosmograph-render-components", type=int, default=1)
    parser.add_argument("--cosmograph-render-max-nodes", type=int, default=500)
    parser.add_argument(
        "--cosmograph-render-labels",
        type=int,
        default=75,
        help="Number of top labels to show in the Cosmograph candidate.",
    )
    parser.add_argument("--cosmograph-render-width", type=int, default=3200)
    parser.add_argument("--cosmograph-render-height", type=int, default=1900)
    parser.add_argument("--cosmograph-render-wait-ms", type=int, default=20000)
    parser.add_argument(
        "--use-browser-screenshots",
        action="store_true",
        help="Use browser screenshots for ECharts PNGs. Static PNG rendering is the default for manuscript figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
    run(parse_args())
