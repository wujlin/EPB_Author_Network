"""
EPB

"""

import pandas as pd
import numpy as np
import json
import networkx as nx
from collections import defaultdict
import os
from typing import Dict, List, Set, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality font and style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

#  - 
FONT_CONFIG = {
    'title': 22,      # 
    'label': 18,      # xlabel, ylabel
    'tick': 16,       # 
    'legend': 14,     # 
    'suptitle': 24    # 
}
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5

# Publication-quality color palettes
JOURNAL_COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Deep magenta
    'accent': '#F18F01',       # Vibrant orange
    'success': '#C73E1D',      # Deep red
    'neutral': '#6C757D',      # Professional gray
    'light': '#E9ECEF',        # Light gray
    'dark': '#212529'          # Dark gray
}

NATURE_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

SCIENCE_PALETTE = ['#4472C4', '#E70000', '#70AD47', '#FFC000', '#5B9BD5', 
                   '#A5A5A5', '#264478', '#9C0006', '#548235', '#B2900F']

class EPBNetworkBuilder:
    """EPB"""
    
    def __init__(self, author_mapping_file: str, data_directory: str):
        """
        
        
        Args:
            author_mapping_file: 
            data_directory: 
        """
        self.author_mapping_file = author_mapping_file
        self.data_directory = data_directory
        self.author_mapping = {}
        self.name_to_canonical = {}
        self.papers_data = pd.DataFrame()
        self.collaboration_network = nx.Graph()
        
        print(" EPB...")
        self._load_author_mapping()
        self._load_papers_data()
    
    def _load_author_mapping(self):
        """"""
        print(" ...")
        
        with open(self.author_mapping_file, 'r', encoding='utf-8') as f:
            self.author_mapping = json.load(f)
        
        #  -> 
        for canonical, variants in self.author_mapping.items():
            for variant in variants:
                self.name_to_canonical[variant] = canonical
        
        print(f" : {len(self.author_mapping)} , {len(self.name_to_canonical)} ")
    
    def _load_papers_data(self):
        """"""
        print(" ...")
        
        # 
        data_files = []
        for decade in ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s', 
        "article_24feb", 'EPB_Data', 'EPB_FG', 'featured graphics', 'inpress_add',
        "online", "urban data code"]:
            file_path = os.path.join(self.data_directory, f'{decade}.csv')
            if os.path.exists(file_path):
                data_files.append(file_path)
        
        all_papers = []
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                decade = os.path.basename(file_path).replace('.csv', '')
                df['Decade'] = decade
                all_papers.append(df)
                print(f"    {decade}: {len(df)} ")
            except Exception as e:
                print(f"    {file_path}: {e}")
        
        if all_papers:
            self.papers_data = pd.concat(all_papers, ignore_index=True)
            print(f" : {len(self.papers_data)} ")
        else:
            raise ValueError("")
    
    def _split_authors(self, author_string: str) -> List[str]:
        """
         "Last, First" 
        : "Johnson, Daniel; See, Linda; Oswald, Sandro M"
        : ["Johnson, Daniel", "See, Linda", "Oswald, Sandro M"]
        """
        if pd.isna(author_string) or not isinstance(author_string, str):
            return []
        
        author_string = author_string.strip()
        
        #  "Last, First" 
        if ';' in author_string:
            authors = [author.strip() for author in author_string.split(';')]
        elif '|' in author_string:
            authors = [author.strip() for author in author_string.split('|')]
        elif ' and ' in author_string:
            authors = [author.strip() for author in author_string.split(' and ')]
        elif ' & ' in author_string:
            authors = [author.strip() for author in author_string.split(' & ')]
        else:
            # 
            authors = [author_string]
        
        # 
        return [author for author in authors if author and len(author) > 2]
    
    def _standardize_author_name(self, author_name: str) -> str:
        """"""
        author_name = author_name.strip()
        return self.name_to_canonical.get(author_name, author_name)
    
    def build_collaboration_network(self, use_weighted: bool = True) -> nx.Graph:
        """
        
        
        Args:
            use_weighted:  = 1/
        
        Returns:
            nx.Graph: 
        """
        print(f"\n {'' if use_weighted else ''}...")
        
        collaboration_weights = defaultdict(float)  # float
        collaboration_counts = defaultdict(int)     # 
        author_papers = defaultdict(list)
        
        for idx, row in self.papers_data.iterrows():
            if pd.isna(row.get('Author')):
                continue
            
            # 
            authors = self._split_authors(row['Author'])
            standardized_authors = [self._standardize_author_name(author) for author in authors]
            standardized_authors = list(set(standardized_authors))  # 
            
            # 
            paper_info = {
                'title': row.get('Title', ''),
                'year': row.get('Publication Year', ''),
                'decade': row.get('Decade', ''),
                'journal': row.get('Source title', ''),
                'index': idx,
                'n_authors': len(standardized_authors)  # 
            }
            
            for author in standardized_authors:
                author_papers[author].append(paper_info)
            
            # 
            if len(standardized_authors) > 1:
                # 
                paper_weight = 1.0 / len(standardized_authors) if use_weighted else 1.0
                
                for i in range(len(standardized_authors)):
                    for j in range(i + 1, len(standardized_authors)):
                        author1, author2 = standardized_authors[i], standardized_authors[j]
                        
                        # 
                        edge = tuple(sorted([author1, author2]))
                        
                        collaboration_weights[edge] += paper_weight
                        collaboration_counts[edge] += 1
        
        # 
        self.collaboration_network = nx.Graph()
        
        # 
        for author, papers in author_papers.items():
            # 
            paper_types = {}
            for paper in papers:
                n_authors = paper['n_authors']
                paper_types[n_authors] = paper_types.get(n_authors, 0) + 1
            
            self.collaboration_network.add_node(author, 
                                               paper_count=len(papers),
                                               papers=papers,
                                               paper_types=paper_types)
        
        # 
        total_weight = 0
        for edge, weight in collaboration_weights.items():
            author1, author2 = edge
            count = collaboration_counts[edge]
            
            self.collaboration_network.add_edge(author1, author2, 
                                               weight=weight,
                                               count=count,
                                               avg_weight=weight/count)
            total_weight += weight
        
        print(f" :")
        print(f"  - : {self.collaboration_network.number_of_nodes()}")
        print(f"  - : {self.collaboration_network.number_of_edges()}")
        if use_weighted:
            print(f"  - : {total_weight:.2f}")
            print(f"  - : {total_weight/self.collaboration_network.number_of_edges():.4f}")
        
        return self.collaboration_network
    
    def get_network_basic_info(self) -> Dict:
        """
        
        
        Returns:
            Dict: 
        """
        if self.collaboration_network.number_of_nodes() == 0:
            print(" ")
            return {}
        
        basic_info = {
            'nodes': self.collaboration_network.number_of_nodes(),
            'edges': self.collaboration_network.number_of_edges(),
            'density': nx.density(self.collaboration_network),
            'connected_components': nx.number_connected_components(self.collaboration_network)
        }
        
        # 
        if basic_info['connected_components'] > 0:
            largest_cc = max(nx.connected_components(self.collaboration_network), key=len)
            basic_info['largest_component_size'] = len(largest_cc)
            basic_info['largest_component_ratio'] = len(largest_cc) / basic_info['nodes']
        
        return basic_info
    
    def get_author_info(self, author_name: str) -> Dict:
        """
        
        
        Args:
            author_name: 
            
        Returns:
            Dict: 
        """
        # 
        canonical_name = self._standardize_author_name(author_name)
        
        if canonical_name not in self.collaboration_network.nodes:
            return {"error": f" '{author_name}' "}
        
        node_data = self.collaboration_network.nodes[canonical_name]
        neighbors = list(self.collaboration_network.neighbors(canonical_name))
        
        author_info = {
            'canonical_name': canonical_name,
            'paper_count': node_data.get('paper_count', 0),
            'collaboration_count': len(neighbors),
            'collaborators': neighbors,
            'papers': node_data.get('papers', [])
        }
        
        return author_info
    
    def get_collaboration_info(self, author1: str, author2: str) -> Dict:
        """
        
        
        Args:
            author1: 1
            author2: 2
            
        Returns:
            Dict: 
        """
        canonical_name1 = self._standardize_author_name(author1)
        canonical_name2 = self._standardize_author_name(author2)
        
        if not self.collaboration_network.has_edge(canonical_name1, canonical_name2):
            return {"error": f"'{author1}'  '{author2}' "}
        
        edge_data = self.collaboration_network[canonical_name1][canonical_name2]
        
        collaboration_info = {
            'author1': canonical_name1,
            'author2': canonical_name2,
            'collaboration_count': edge_data.get('weight', 1)
        }
        
        return collaboration_info
    
    def save_network(self, filename: str = "epb_collaboration_network.gexf"):
        """
        
        
        Args:
            filename: 
        """
        if self.collaboration_network.number_of_nodes() == 0:
            print(" ")
            return
        
        nx.write_gexf(self.collaboration_network, filename)
        print(f" : {filename}")
    
    def load_network(self, filename: str):
        """
        
        
        Args:
            filename: 
        """
        try:
            self.collaboration_network = nx.read_gexf(filename)
            print(f"  {filename} ")
            print(f"  - : {self.collaboration_network.number_of_nodes()}")
            print(f"  - : {self.collaboration_network.number_of_edges()}")
        except Exception as e:
            print(f" : {e}")

class NetworkAnalyzer:
    """ - """
    
    def __init__(self, network: nx.Graph):
        """
        
        
        Args:
            network: 
        """
        self.network = network
        self.filtered_network = None
        
    def analyze_degree_distribution(self, use_weighted: bool = True) -> Dict:
        """
        
        
        Args:
            use_weighted: 
        
        Returns:
            Dict: 
        """
        print(f" {'' if use_weighted else ''}...")
        
        if use_weighted:
            # 
            degrees = [sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True)) 
                      for node in self.network.nodes()]
        else:
            # 
            degrees = [d for n, d in self.network.degree()]
        
        if not degrees:
            return {}
        
        stats = {
            'type': 'weighted_degree' if use_weighted else 'unweighted_degree',
            'min_degree': min(degrees),
            'max_degree': max(degrees),
            'mean_degree': np.mean(degrees),
            'median_degree': np.median(degrees),
            'std_degree': np.std(degrees),
            'degree_distribution': pd.Series(degrees).value_counts().sort_index()
        }
        
        print(f"{'' if use_weighted else ''}:")
        print(f"  - : {stats['min_degree']:.4f}")
        print(f"  - : {stats['max_degree']:.4f}")
        print(f"  - : {stats['mean_degree']:.4f}")
        print(f"  - : {stats['median_degree']:.4f}")
        print(f"  - : {stats['std_degree']:.4f}")
        
        return stats
    
    def filter_network_by_degree(self, min_degree: float = 5.0, max_nodes: int = 500, use_weighted: bool = True) -> nx.Graph:
        """
        Filter network nodes based on degree threshold (supports weighted and unweighted degree)
        
        Args:
            min_degree: Minimum degree threshold
                       - If use_weighted=True: minimum weighted degree (sum of edge weights)
                       - If use_weighted=False: minimum unweighted degree (number of neighbors)
            max_nodes: Maximum number of nodes to retain
            use_weighted: Whether to use weighted degree for filtering
                         - True: Filter by weighted degree ( edge_weights)
                         - False: Filter by unweighted degree (neighbor count)
            
        Returns:
            nx.Graph: Filtered network subgraph
            
        Example:
            # Filter nodes with weighted degree  3.0
            filtered_net = analyzer.filter_network_by_degree(min_degree=3.0, use_weighted=True)
            
            # Filter nodes with  5 neighbors  
            filtered_net = analyzer.filter_network_by_degree(min_degree=5, use_weighted=False)
        """
        degree_type = "weighted degree" if use_weighted else "unweighted degree"
        print(f" Filtering network: minimum {degree_type} >= {min_degree}, max nodes <= {max_nodes}")
        
        # 
        if use_weighted:
            # 
            degree_dict = {}
            for node in self.network.nodes():
                weighted_degree = sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True))
                degree_dict[node] = weighted_degree
        else:
            # 
            degree_dict = dict(self.network.degree())
        
        # min_degree
        high_degree_nodes = [n for n, d in degree_dict.items() if d >= min_degree]
        
        print(f"  Nodes with {degree_type} >= {min_degree}: {len(high_degree_nodes)}")
        
        # If still too many nodes, select the top max_nodes by degree
        if len(high_degree_nodes) > max_nodes:
            high_degree_nodes = sorted(high_degree_nodes, 
                                     key=lambda x: degree_dict[x], 
                                     reverse=True)[:max_nodes]
            print(f"  Further filtered to top {len(high_degree_nodes)} nodes by {degree_type}")
        
        # 
        self.filtered_network = self.network.subgraph(high_degree_nodes).copy()
        
        # min_degree>0
        if min_degree > 0:
            isolated_nodes = [n for n in self.filtered_network.nodes() if self.filtered_network.degree(n) == 0]
            if isolated_nodes:
                print(f"   Found {len(isolated_nodes)} isolated nodes after subgraph creation")
                print(f"   Removing isolated nodes to improve visualization...")
                self.filtered_network.remove_nodes_from(isolated_nodes)
        
        print(f" Filtering completed:")
        print(f"  - Nodes: {self.filtered_network.number_of_nodes()}")
        print(f"  - Edges: {self.filtered_network.number_of_edges()}")
        print(f"  - Isolated nodes: {len([n for n in self.filtered_network.nodes() if self.filtered_network.degree(n) == 0])}")
        print(f"  - Density: {nx.density(self.filtered_network):.6f}")
        
        return self.filtered_network
    
    def plot_degree_distribution(self, save_path: str = None):
        """
        
        
        Args:
            save_path: 
        """
        degrees = [d for n, d in self.network.degree()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 
        ax1.hist(degrees, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Degree', fontweight='bold', fontsize=FONT_CONFIG['label'])
        ax1.set_ylabel('Frequency', fontweight='bold', fontsize=FONT_CONFIG['label'])
        ax1.set_title('Degree Distribution', fontweight='bold', fontsize=FONT_CONFIG['title'])
        ax1.grid(True, alpha=0.3)
        
        # 
        sorted_degrees = sorted(degrees, reverse=True)
        ax2.plot(range(1, len(sorted_degrees) + 1), sorted_degrees, 'b-', linewidth=2)
        ax2.set_xlabel('Rank', fontweight='bold', fontsize=FONT_CONFIG['label'])
        ax2.set_ylabel('Degree', fontweight='bold', fontsize=FONT_CONFIG['label'])
        ax2.set_title('Degree Ranking', fontweight='bold', fontsize=FONT_CONFIG['title'])
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" : {save_path}")
        
        plt.show()
    
    def visualize_network(self, 
                         network: nx.Graph = None,
                         layout_type: str = 'spring',
                         node_size_attr: str = 'paper_count',
                         edge_width_attr: str = 'weight',
                         show_labels: bool = True,
                         label_threshold: int = 10,
                         figsize: tuple = (15, 12),
                         save_path: str = None):
        """
        
        
        Args:
            network: None
            layout_type:  ('spring', 'circular', 'kamada_kawai', 'fruchterman_reingold')
            node_size_attr: 
            edge_width_attr: 
            show_labels: 
            label_threshold: 
            figsize: 
            save_path: 
        """
        if network is None:
            if self.filtered_network is None:
                print(" ")
                return
            network = self.filtered_network
        
        print(f" : {network.number_of_nodes()} , {network.number_of_edges()} ")
        
        # 
        if layout_type == 'spring':
            pos = nx.spring_layout(network, k=2, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(network)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(network)
        elif layout_type == 'fruchterman_reingold':
            pos = nx.fruchterman_reingold_layout(network)
        else:
            pos = nx.spring_layout(network)
        
        plt.figure(figsize=figsize)
        
        # 
        if node_size_attr in ['paper_count', 'degree']:
            if node_size_attr == 'paper_count':
                node_sizes = [network.nodes[node].get('paper_count', 1) * 20 
                             for node in network.nodes()]
            else:  # degree
                node_sizes = [network.degree(node) * 10 for node in network.nodes()]
        else:
            node_sizes = [50] * network.number_of_nodes()
        
        # 
        if edge_width_attr == 'weight':
            edge_widths = [network[u][v].get('weight', 1) * 0.5 
                          for u, v in network.edges()]
        else:
            edge_widths = [1] * network.number_of_edges()
        
        # 
        nx.draw_networkx_edges(network, pos, 
                              width=edge_widths, 
                              alpha=0.3, 
                              edge_color='gray')
        
        # 
        nx.draw_networkx_nodes(network, pos, 
                              node_size=node_sizes,
                              node_color='lightblue',
                              alpha=0.8,
                              linewidths=1,
                              edgecolors='navy')
        
        # 
        if show_labels:
            high_degree_nodes = [n for n in network.nodes() 
                               if network.degree(n) >= label_threshold]
            
            if high_degree_nodes:
                # 
                labels = {}
                for node in high_degree_nodes:
                    labels[node] = node  # 
                
                nx.draw_networkx_labels(network, pos, 
                                      labels=labels,
                                      font_size=8,
                                      font_weight='bold',
                                      bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor="white", 
                                               alpha=0.7))
        
        plt.title(f"EPB Journal Author Collaboration Network\n({network.number_of_nodes()} Authors, {network.number_of_edges()} Collaborations)", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" : {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def get_top_authors_by_metric(self, metric: str = 'weighted_degree', top_n: int = 20) -> List[Tuple[str, float]]:
        """
        
        
        Args:
            metric:  ('weighted_degree', 'degree', 'betweenness', 'closeness', 'eigenvector', 'paper_count')
            top_n: N
            
        Returns:
            List[Tuple[str, float]]: (, ) 
        """
        print(f"  (: {metric})")
        
        if metric == 'weighted_degree':
            # 
            scores = {}
            for node in self.network.nodes():
                weighted_degree = sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True))
                scores[node] = weighted_degree
        elif metric == 'degree':
            scores = dict(self.network.degree())
        elif metric == 'paper_count':
            scores = {node: data.get('paper_count', 0) 
                     for node, data in self.network.nodes(data=True)}
        elif metric == 'betweenness':
            # 
            largest_cc = max(nx.connected_components(self.network), key=len)
            subgraph = self.network.subgraph(largest_cc)
            scores = nx.betweenness_centrality(subgraph, weight='weight')
        elif metric == 'closeness':
            largest_cc = max(nx.connected_components(self.network), key=len)
            subgraph = self.network.subgraph(largest_cc)
            scores = nx.closeness_centrality(subgraph, distance='weight')
        elif metric == 'eigenvector':
            try:
                largest_cc = max(nx.connected_components(self.network), key=len)
                subgraph = self.network.subgraph(largest_cc)
                scores = nx.eigenvector_centrality(subgraph, weight='weight', max_iter=1000)
            except:
                print(" ")
                scores = {}
                for node in self.network.nodes():
                    weighted_degree = sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True))
                    scores[node] = weighted_degree
        else:
            raise ValueError(f": {metric}")
        
        # N
        top_authors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"{top_n} ({metric}):")
        for i, (author, score) in enumerate(top_authors, 1):
            paper_count = self.network.nodes[author].get('paper_count', 0) if author in self.network.nodes else 0
            if metric == 'weighted_degree':
                # 
                unweighted_degree = self.network.degree(author) if author in self.network else 0
                print(f"  {i:2d}. {author}: {score:.4f} (:{unweighted_degree}, {paper_count})")
            else:
                print(f"  {i:2d}. {author}: {score:.4f} ({paper_count})")
        
        return top_authors
    
    def analyze_network_structure(self) -> Dict:
        """
        
        
        Returns:
            Dict: 
        """
        print(" ...")
        
        # 
        results = {
            'nodes': self.network.number_of_nodes(),
            'edges': self.network.number_of_edges(),
            'density': nx.density(self.network),
            'connected_components': nx.number_connected_components(self.network)
        }
        
        # 
        cc_sizes = [len(cc) for cc in nx.connected_components(self.network)]
        results['largest_component_size'] = max(cc_sizes) if cc_sizes else 0
        results['largest_component_ratio'] = results['largest_component_size'] / results['nodes']
        
        # 
        largest_cc = max(nx.connected_components(self.network), key=len)
        largest_subgraph = self.network.subgraph(largest_cc)
        results['average_clustering'] = nx.average_clustering(largest_subgraph)
        
        # 
        if len(largest_cc) > 1 and len(largest_cc) < 1000:  # 
            try:
                results['average_shortest_path'] = nx.average_shortest_path_length(largest_subgraph)
            except:
                results['average_shortest_path'] = None
        else:
            results['average_shortest_path'] = None
        
        print(":")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.6f}")
            else:
                print(f"  - {key}: {value}")
        
        return results
    
    def diagnose_name_issues(self, top_n: int = 30) -> Dict:
        """
        
        
        Args:
            top_n: N
            
        Returns:
            Dict: 
        """
        print(" ...")
        
        # 
        degree_dict = dict(self.network.degree())
        top_authors = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        problematic_cases = []
        
        print(f"\n{'':<4} {'':<20} {'':<8} {'':<8} {''}")
        print("-" * 60)
        
        for i, (author, degree) in enumerate(top_authors, 1):
            paper_count = self.network.nodes[author].get('paper_count', 0)
            issues = []
            
            # /
            if ',' not in author and ' ' not in author:
                # 
                chinese_surnames = ['Wang', 'Zhang', 'Liu', 'Chen', 'Yang', 'Huang', 'Gao', 'Li', 'Wu', 'Zhou', 'Xu', 'Lin', 'He', 'Ma', 'Sun', 'Zhao', 'Zhu', 'Luo', 'Han']
                # 
                western_names = ['Michael', 'David', 'Richard', 'Mark', 'Daniel', 'Martin', 'Paul', 'John', 'James', 'Robert', 'William', 'Thomas', 'Christopher', 'Matthew', 'Anthony', 'Donald', 'Steven', 'Kenneth', 'Joshua', 'Kevin', 'Brian', 'George', 'Timothy', 'Ronald', 'Jason', 'Edward', 'Jeffrey', 'Ryan', 'Jacob', 'Gary', 'Nicholas', 'Eric', 'Jonathan', 'Stephen', 'Larry', 'Justin', 'Scott', 'Brandon', 'Benjamin', 'Samuel', 'Gregory', 'Alexander', 'Patrick', 'Frank', 'Raymond', 'Jack', 'Dennis', 'Jerry', 'Tyler', 'Aaron', 'Jose', 'Henry', 'Adam', 'Douglas', 'Nathan', 'Peter', 'Zachary', 'Kyle', 'Walter', 'Harold']
                
                if author in chinese_surnames:
                    issues.append(" ")
                    problematic_cases.append({
                        'name': author,
                        'type': 'chinese_surname_merge',
                        'degree': degree,
                        'paper_count': paper_count,
                        'severity': 'critical'
                    })
                elif author in western_names:
                    issues.append(" ")
                    problematic_cases.append({
                        'name': author,
                        'type': 'western_name_merge',
                        'degree': degree,
                        'paper_count': paper_count,
                        'severity': 'critical'
                    })
                else:
                    issues.append(" ")
            
            # 
            if paper_count > 0:
                ratio = degree / paper_count
                if ratio > 10:  # 
                    issues.append(f"  (:{ratio:.1f})")
            
            if not issues:
                issues.append(" ")
            
            issue_text = "; ".join(issues)
            print(f"{i:<4} {author:<20} {degree:<8} {paper_count:<8} {issue_text}")
        
        # 
        critical_issues = [case for case in problematic_cases if case['severity'] == 'critical']
        
        print(f"\n :")
        print(f"  - : {len([c for c in critical_issues if c['type'] == 'chinese_surname_merge'])}")
        print(f"  - : {len([c for c in critical_issues if c['type'] == 'western_name_merge'])}")
        print(f"  - : {len(critical_issues)}")
        
        if critical_issues:
            print(f"\n  ")
            print(f"    _split_authors  'Last, First' ")
        
        return {
            'problematic_cases': problematic_cases,
            'critical_count': len(critical_issues),
            'top_authors': top_authors
        }

    def analyze_weight_distribution(self) -> Dict:
        """
        
        
        Returns:
            Dict: 
        """
        print(" ...")
        
        # 
        weights = [data.get('weight', 1) for _, _, data in self.network.edges(data=True)]
        counts = [data.get('count', 1) for _, _, data in self.network.edges(data=True)]
        avg_weights = [data.get('avg_weight', 1) for _, _, data in self.network.edges(data=True)]
        
        if not weights:
            return {}
        
        stats = {
            'total_edges': len(weights),
            'weight_stats': {
                'min': min(weights),
                'max': max(weights),
                'mean': np.mean(weights),
                'median': np.median(weights),
                'std': np.std(weights)
            },
            'count_stats': {
                'min': min(counts),
                'max': max(counts),
                'mean': np.mean(counts),
                'median': np.median(counts)
            },
            'avg_weight_stats': {
                'min': min(avg_weights),
                'max': max(avg_weights),
                'mean': np.mean(avg_weights),
                'median': np.median(avg_weights)
            }
        }
        
        print(f":")
        print(f"  : {stats['total_edges']}")
        print(f"  : {stats['weight_stats']['min']:.4f} - {stats['weight_stats']['max']:.4f}")
        print(f"  : {stats['weight_stats']['mean']:.4f}")
        print(f"  : {stats['count_stats']['min']} - {stats['count_stats']['max']}")
        print(f"  : {stats['count_stats']['mean']:.2f}")
        print(f"  : {stats['avg_weight_stats']['min']:.4f} - {stats['avg_weight_stats']['max']:.4f}")
        
        return stats
    
    def analyze_weighted_degree_distribution_detailed(self, plot: bool = True, save_path: str = None) -> Dict:
        """
        
        
        Args:
            plot: 
            save_path: 
            
        Returns:
            Dict: 
        """
        print("\n" + "="*60)
        print(" Detailed Analysis of Weighted Degree Distribution")
        print("="*60)
        
        # Calculate weighted degrees for all nodes
        weighted_degrees = []
        degree_composition = {}  # Store composition analysis for each weighted degree
        
        for node in self.network.nodes():
            # Calculate weighted degree as sum of edge weights
            node_weighted_degree = sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True))
            weighted_degrees.append(node_weighted_degree)
            
            # Analyze composition: what edge weights contribute to this weighted degree
            edge_weights = [data.get('weight', 1) for _, _, data in self.network.edges(node, data=True)]
            edge_counts = [data.get('count', 1) for _, _, data in self.network.edges(node, data=True)]
            
            # Round weighted degree to avoid floating point precision issues
            rounded_wd = round(node_weighted_degree, 4)
            if rounded_wd not in degree_composition:
                degree_composition[rounded_wd] = {
                    'count': 0,
                    'edge_weight_patterns': [],
                    'collaboration_patterns': []
                }
            
            degree_composition[rounded_wd]['count'] += 1
            degree_composition[rounded_wd]['edge_weight_patterns'].append(sorted(edge_weights, reverse=True))
            degree_composition[rounded_wd]['collaboration_patterns'].append(sorted(edge_counts, reverse=True))
        
        weighted_degrees = np.array(weighted_degrees)
        
        # Basic statistics
        stats = {
            'total_nodes': len(weighted_degrees),
            'min_degree': np.min(weighted_degrees),
            'max_degree': np.max(weighted_degrees),
            'mean_degree': np.mean(weighted_degrees),
            'median_degree': np.median(weighted_degrees),
            'std_degree': np.std(weighted_degrees),
            'unique_values': len(np.unique(weighted_degrees))
        }
        
        # Analyze distribution characteristics
        degree_counts = pd.Series(weighted_degrees).value_counts().sort_index()
        
        print(f"Basic Statistics:")
        print(f"  Total nodes: {stats['total_nodes']:,}")
        print(f"  Weighted degree range: {stats['min_degree']:.4f} - {stats['max_degree']:.4f}")
        print(f"  Mean weighted degree: {stats['mean_degree']:.4f}")
        print(f"  Median weighted degree: {stats['median_degree']:.4f}")
        print(f"  Standard deviation: {stats['std_degree']:.4f}")
        print(f"  Unique weighted degree values: {stats['unique_values']:,}")
        
        # Find most frequent weighted degree values
        most_frequent = degree_counts.head(10)
        print(f"\nTop 10 Most Frequent Weighted Degree Values:")
        for degree_val, count in most_frequent.items():
            percentage = (count / len(weighted_degrees)) * 100
            print(f"  {degree_val:.4f}: {count:,} nodes ({percentage:.2f}%)")
        
        # Analyze why certain values are so frequent
        print(f"\n Analysis of High-Frequency Weighted Degrees:")
        
        # Focus on the most frequent values
        top_3_degrees = most_frequent.head(3).index.tolist()
        
        for degree_val in top_3_degrees:
            rounded_degree = round(degree_val, 4)
            if rounded_degree in degree_composition:
                comp_data = degree_composition[rounded_degree]
                print(f"\nWeighted Degree = {degree_val:.4f} ({comp_data['count']} nodes):")
                
                # Analyze edge weight patterns
                edge_patterns = comp_data['edge_weight_patterns']
                pattern_counts = {}
                for pattern in edge_patterns:
                    # Convert to tuple for hashing, keep only first few significant weights
                    pattern_key = tuple(round(w, 4) for w in pattern[:5])  # Top 5 weights
                    pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
                
                print(f"  Most common edge weight patterns:")
                sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
                for i, (pattern, count) in enumerate(sorted_patterns[:3]):
                    percentage = (count / len(edge_patterns)) * 100
                    pattern_str = " + ".join([f"{w:.4f}" for w in pattern])
                    print(f"    Pattern {i+1}: {pattern_str} ({count} nodes, {percentage:.1f}%)")
                
                # Analyze collaboration patterns
                collab_patterns = comp_data['collaboration_patterns']
                collab_counts = {}
                for pattern in collab_patterns:
                    pattern_key = tuple(pattern[:5])  # Top 5 collaboration counts
                    collab_counts[pattern_key] = collab_counts.get(pattern_key, 0) + 1
                
                print(f"  Most common collaboration patterns:")
                sorted_collab = sorted(collab_counts.items(), key=lambda x: x[1], reverse=True)
                for i, (pattern, count) in enumerate(sorted_collab[:3]):
                    percentage = (count / len(collab_patterns)) * 100
                    pattern_str = " + ".join([f"{c}" for c in pattern])
                    print(f"    Pattern {i+1}: {pattern_str} collaborations ({count} nodes, {percentage:.1f}%)")
        
        # Collaboration Strategy Analysis - moved from visualization to logs
        print(f"\n Collaboration Strategy Analysis:")
        print(f"  Analyzing breadth vs intensity patterns in EPB collaboration network...")
        
        # Calculate collaboration metrics for all nodes (for logging)
        breadth_all = []
        intensity_all = []
        strength_all = []
        
        for node in self.network.nodes():
            neighbors = list(self.network.neighbors(node))
            breadth = len(neighbors)
            total_strength = sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True))
            intensity = total_strength / breadth if breadth > 0 else 0
            
            breadth_all.append(breadth)
            intensity_all.append(intensity)
            strength_all.append(total_strength)
        
        # Filter out isolated nodes for analysis
        active_mask = np.array(breadth_all) > 0
        breadth_active = np.array(breadth_all)[active_mask]
        intensity_active = np.array(intensity_all)[active_mask]
        strength_active = np.array(strength_all)[active_mask]
        
        if len(breadth_active) > 0:
            # Calculate key statistics
            median_breadth = np.median(breadth_active)
            median_intensity = np.median(intensity_active)
            mean_breadth = np.mean(breadth_active)
            mean_intensity = np.mean(intensity_active)
            correlation = np.corrcoef(breadth_active, intensity_active)[0, 1]
            
            # Identify collaboration strategy quadrants
            high_breadth_mask = breadth_active >= median_breadth
            high_intensity_mask = intensity_active >= median_intensity
            
            super_collaborators = np.sum(high_breadth_mask & high_intensity_mask)
            intensive_partners = np.sum(~high_breadth_mask & high_intensity_mask)
            extensive_networkers = np.sum(high_breadth_mask & ~high_intensity_mask)
            moderate_collaborators = np.sum(~high_breadth_mask & ~high_intensity_mask)
            
            # Top collaborators analysis
            top_threshold = np.percentile(strength_active, 95)
            top_collaborators = np.sum(strength_active >= top_threshold)
            
            print(f"\n   Network Statistics:")
            print(f"    Active scholars: {len(breadth_active):,}")
            print(f"    Collaboration breadth: {mean_breadth:.1f}  {np.std(breadth_active):.1f} (median: {median_breadth:.1f})")
            print(f"    Collaboration intensity: {mean_intensity:.3f}  {np.std(intensity_active):.3f} (median: {median_intensity:.3f})")
            print(f"    Breadth-Intensity correlation: r = {correlation:.3f}")
            
            print(f"\n   Collaboration Strategy Distribution:")
            total_active = len(breadth_active)
            print(f"    Super Collaborators (High Breadth + High Intensity): {super_collaborators:4d} ({super_collaborators/total_active*100:.1f}%)")
            print(f"    Intensive Partners (Low Breadth + High Intensity):   {intensive_partners:4d} ({intensive_partners/total_active*100:.1f}%)")
            print(f"    Extensive Networkers (High Breadth + Low Intensity): {extensive_networkers:4d} ({extensive_networkers/total_active*100:.1f}%)")
            print(f"    Moderate Collaborators (Low Breadth + Low Intensity): {moderate_collaborators:4d} ({moderate_collaborators/total_active*100:.1f}%)")
            
            print(f"\n   Elite Collaboration Analysis:")
            print(f"    Top 5% collaborators: {top_collaborators} scholars")
            print(f"    Elite threshold: Total weighted degree  {top_threshold:.2f}")
            
            print(f"\n   Strategic Insights:")
            if correlation > 0.3:
                print(f"    - POSITIVE correlation: Scholars tend to balance breadth and intensity")
            elif correlation < -0.3:
                print(f"    - NEGATIVE correlation: Trade-off between breadth and intensity")
            else:
                print(f"    - WEAK correlation: Diverse collaboration strategies coexist")
            
            if super_collaborators/total_active > 0.3:
                print(f"    - High proportion of super collaborators indicates mature collaboration network")
            if intensive_partners/total_active > 0.3:
                print(f"    - Strong intensive partnership culture in EPB network")
            if extensive_networkers/total_active > 0.3:
                print(f"    - Significant presence of extensive networking behavior")
        
        if plot:
            fig = plt.figure(figsize=(20, 12))
            
            # Create a 2x2 subplot layout (df4)
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # 1. Histogram of weighted degrees
            ax1 = fig.add_subplot(gs[0, 0])
            n, bins, patches = ax1.hist(weighted_degrees, bins=50, alpha=0.8, 
                                      color=JOURNAL_COLORS['primary'], edgecolor='white', linewidth=1)
            
            # Color gradient
            for i, p in enumerate(patches):
                p.set_facecolor(plt.cm.Blues(0.3 + 0.7 * n[i] / max(n)))
            
            ax1.axvline(x=stats['mean_degree'], color=JOURNAL_COLORS['accent'], 
                       linestyle='--', linewidth=2, label=f'Mean = {stats["mean_degree"]:.3f}')
            ax1.axvline(x=stats['median_degree'], color=JOURNAL_COLORS['success'], 
                       linestyle='-', linewidth=2, label=f'Median = {stats["median_degree"]:.3f}')
            
            ax1.set_xlabel('Weighted Degree', fontweight='bold', fontsize=FONT_CONFIG['label'])
            ax1.set_ylabel('Frequency', fontweight='bold', fontsize=FONT_CONFIG['label'])
            ax1.set_title('a. Weighted Degree Distribution', fontweight='bold', fontsize=FONT_CONFIG['title'])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Top frequent values bar chart
            ax2 = fig.add_subplot(gs[0, 1])
            top_10 = most_frequent.head(10)
            bars = ax2.bar(range(len(top_10)), top_10.values, 
                          color=SCIENCE_PALETTE[:len(top_10)], alpha=0.8, edgecolor='white')
            
            ax2.set_xlabel('Rank', fontweight='bold', fontsize=FONT_CONFIG['label'])
            ax2.set_ylabel('Number of Nodes', fontweight='bold', fontsize=FONT_CONFIG['label'])
            ax2.set_title('b. Top 10 Most Frequent Values', fontweight='bold', fontsize=FONT_CONFIG['title'])
            ax2.set_xticks(range(len(top_10)))
            ax2.set_xticklabels([f'{v:.3f}' for v in top_10.index], rotation=45, ha='right', fontsize=FONT_CONFIG['tick'])
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            # for bar, value in zip(bars, top_10.values):
            #     height = bar.get_height()
            #     ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            #                 f'{value:,}', ha='center', va='bottom', fontsize=FONT_CONFIG['tick'])
            
            # 3. Log-log plot to examine power law ([1,0]2x2)
            ax3 = fig.add_subplot(gs[1, 0])
            x = degree_counts.index.values
            y = degree_counts.values
            
            # Filter positive values for log transformation
            mask = (x > 0) & (y > 0)
            x_filtered = x[mask]
            y_filtered = y[mask]
            
            if len(x_filtered) > 0:
                scatter = ax3.loglog(x_filtered, y_filtered, 'o', 
                                   color=JOURNAL_COLORS['secondary'], markersize=6, alpha=0.7,
                                   markeredgecolor='white', markeredgewidth=1)
                
                # Highlight y=1 line (frequency = 1)
                ax3.axhline(y=1, color=JOURNAL_COLORS['neutral'], linestyle='--', 
                          alpha=0.8, linewidth=2, label='Frequency = 1')
                
                # Add power law fit if enough points
                if len(x_filtered) > 5:
                    log_x = np.log10(x_filtered)
                    log_y = np.log10(y_filtered)
                    z = np.polyfit(log_x, log_y, 1)
                    fit_line = 10**(z[1]) * x_filtered**z[0]
                    ax3.loglog(x_filtered, fit_line, '--', 
                              color=JOURNAL_COLORS['accent'], linewidth=2.5, alpha=0.9,
                              label=f'Power law fit ( = {-z[0]:.2f})')
                
                ax3.set_xlabel('log(Weighted Degree)', fontweight='bold', fontsize=FONT_CONFIG['label'])
                ax3.set_ylabel('log(Frequency)', fontweight='bold', fontsize=FONT_CONFIG['label'])
                ax3.set_title('c. Power-law Analysis', fontweight='bold', fontsize=FONT_CONFIG['title'])
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. Cumulative distribution ( - )
            # ax4 = fig.add_subplot(gs[1, 0])
            # sorted_degrees = np.sort(weighted_degrees)
            # cumulative = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
            # 
            # ax4.plot(sorted_degrees, cumulative, color=JOURNAL_COLORS['primary'], 
            #         linewidth=2, alpha=0.8)
            # ax4.axhline(y=0.5, color=JOURNAL_COLORS['success'], linestyle='--', 
            #            alpha=0.7, label='50th percentile')
            # ax4.axhline(y=0.9, color=JOURNAL_COLORS['accent'], linestyle='--', 
            #            alpha=0.7, label='90th percentile')
            # 
            # ax4.set_xlabel('Weighted Degree', fontweight='bold')
            # ax4.set_ylabel('Cumulative Probability', fontweight='bold')
            # ax4.set_title('D. Cumulative Distribution', fontweight='bold', fontsize=14)
            # ax4.legend()
            # ax4.grid(True, alpha=0.3)
            
            # 4. Collaboration Intensity vs Breadth Analysis ([1,1]df)
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Calculate collaboration breadth and intensity for each node
            breadth_values = []  # Number of unique collaborators
            intensity_values = []  # Average collaboration strength per partner
            total_strength_values = []  # Total weighted degree for color coding
            
            for node in self.network.nodes():
                # Collaboration breadth: number of unique collaborators
                neighbors = list(self.network.neighbors(node))
                breadth = len(neighbors)
                
                # Total collaboration strength (weighted degree)
                total_strength = sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True))
                
                # Average collaboration intensity per partner
                intensity = total_strength / breadth if breadth > 0 else 0
                
                breadth_values.append(breadth)
                intensity_values.append(intensity)
                total_strength_values.append(total_strength)
            
            # Convert to numpy arrays for easier manipulation
            breadth_array = np.array(breadth_values)
            intensity_array = np.array(intensity_values)
            strength_array = np.array(total_strength_values)
            
            # Filter out isolated nodes for better visualization
            mask = breadth_array > 0
            breadth_filtered = breadth_array[mask]
            intensity_filtered = intensity_array[mask]
            strength_filtered = strength_array[mask]
            
            if len(breadth_filtered) > 0:
                # Reduce point density by sampling for better visualization
                n_points = len(breadth_filtered)
                if n_points > 1000:
                    # Sample points but keep all top collaborators
                    sample_size = 800
                    top_strength_threshold = np.percentile(strength_filtered, 95)
                    top_mask = strength_filtered >= top_strength_threshold
                    
                    # Keep all top collaborators
                    top_indices = np.where(top_mask)[0]
                    remaining_indices = np.where(~top_mask)[0]
                    
                    # Sample from remaining points
                    if len(remaining_indices) > sample_size:
                        sampled_remaining = np.random.choice(remaining_indices, 
                                                           size=sample_size, replace=False)
                        plot_indices = np.concatenate([top_indices, sampled_remaining])
                    else:
                        plot_indices = np.arange(n_points)
                    
                    breadth_plot = breadth_filtered[plot_indices]
                    intensity_plot = intensity_filtered[plot_indices]
                    strength_plot = strength_filtered[plot_indices]
                else:
                    breadth_plot = breadth_filtered
                    intensity_plot = intensity_filtered
                    strength_plot = strength_filtered
                    plot_indices = np.arange(n_points)
                
                # Create scatter plot with reduced density and better styling
                scatter = ax4.scatter(breadth_plot, intensity_plot, 
                                    c=strength_plot, 
                                    s=45, alpha=0.6, 
                                    cmap='viridis', 
                                    edgecolors='white', linewidth=0.5,
                                    vmin=np.percentile(strength_filtered, 5),
                                    vmax=np.percentile(strength_filtered, 95))
                
                # Add colorbar with professional styling
                cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8, aspect=20)
                cbar.set_label('Total Weighted Degree', fontweight='bold', fontsize=11)
                cbar.ax.tick_params(labelsize=9)
                
                # Calculate medians for quadrant lines
                median_breadth = np.median(breadth_filtered)
                median_intensity = np.median(intensity_filtered)
                
                # Add subtle quadrant reference lines
                ax4.axvline(x=median_breadth, color='gray', 
                           linestyle='--', alpha=0.4, linewidth=1.0)
                ax4.axhline(y=median_intensity, color='gray', 
                           linestyle='--', alpha=0.4, linewidth=1.0)
                
                # Identify and highlight extreme cases with subtle styling
                top_strength_threshold = np.percentile(strength_filtered, 95)
                top_mask_plot = strength_plot >= top_strength_threshold
                
                if np.any(top_mask_plot):
                    # Highlight top collaborators with subtle orange outline
                    ax4.scatter(breadth_plot[top_mask_plot], intensity_plot[top_mask_plot],
                              s=65, facecolors='none', edgecolors='#FF8C42', 
                              linewidth=1.2, alpha=0.7, 
                              label=f'Top 5% Collaborators (n={np.sum(strength_filtered >= top_strength_threshold)})')
                
                # Customize axes
                ax4.set_xlabel('Collaboration Breadth', 
                              fontweight='bold', fontsize=FONT_CONFIG['label'])
                ax4.set_ylabel('Collaboration Intensity', 
                              fontweight='bold', fontsize=FONT_CONFIG['label'])
                ax4.set_title('d. Collaboration Strategy Analysis', 
                             fontweight='bold', fontsize=FONT_CONFIG['title'])
                
                # Set axis limits with some padding
                max_breadth = np.max(breadth_filtered)
                max_intensity = np.max(intensity_filtered)
                ax4.set_xlim(0, max_breadth * 1.05)
                ax4.set_ylim(0, max_intensity * 1.05)
                
                # Add subtle grid and clean legend
                ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax4.legend(loc='upper right', fontsize=FONT_CONFIG['legend'], framealpha=0.95, 
                          fancybox=True, shadow=True)
                
                # Ensure integer ticks on x-axis (breadth)
                ax4.set_xticks(np.arange(0, int(max_breadth) + 1, max(1, int(max_breadth/8))))
                
            else:
                # Fallback if no valid data
                ax4.text(0.5, 0.5, 'No collaboration data available', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=14, color='gray')
            
            # 6. Edge weight contribution analysis ( - )
            # ax6 = fig.add_subplot(gs[1, 1])
            # 
            # # Analyze what edge weights contribute to top weighted degrees
            # edge_weights_all = []
            # for node in self.network.nodes():
            #     edge_weights_all.extend([data.get('weight', 1) for _, _, data in self.network.edges(node, data=True)])
            # 
            # edge_weight_counts = pd.Series(edge_weights_all).value_counts().sort_index()
            # top_edge_weights = edge_weight_counts.head(10)
            # 
            # bars = ax6.bar(range(len(top_edge_weights)), top_edge_weights.values,
            #               color=plt.cm.viridis(np.linspace(0, 1, len(top_edge_weights))), 
            #               alpha=0.8, edgecolor='white')
            # 
            # ax6.set_xlabel('Edge Weight Value', fontweight='bold')
            # ax6.set_ylabel('Frequency', fontweight='bold') 
            # ax6.set_title('F. Most Common Edge Weights', fontweight='bold', fontsize=14)
            # ax6.set_xticks(range(len(top_edge_weights)))
            # ax6.set_xticklabels([f'{v:.3f}' for v in top_edge_weights.index], 
            #                    rotation=45, ha='right')
            # ax6.grid(True, alpha=0.3, axis='y')
            
            # plt.suptitle('Comprehensive Weighted Degree Distribution Analysis', 
            #             fontsize=20, fontweight='bold', y=0.98)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f" Analysis plot saved to: {save_path}")
            
            plt.tight_layout()
            plt.show()
        
        return {
            'stats': stats,
            'degree_counts': degree_counts,
            'most_frequent': most_frequent,
            'degree_composition': degree_composition,
            'theoretical_explanation': {
                'single_author': 1.0,
                'two_author': 0.5,
                'three_author': 1.0/3,
                'four_author': 0.25,
                'common_combinations': [0.5 + 1.0/3, 0.5 + 0.25, 1.0/3 + 0.25]
            }
        }
    
    def compare_weighted_vs_unweighted(self, top_n: int = 20) -> Dict:
        """
        
        
        Args:
            top_n: N
            
        Returns:
            Dict: 
        """
        print("  vs ...")
        
        # 
        weighted_top = self.get_top_authors_by_metric('weighted_degree', top_n)
        unweighted_top = self.get_top_authors_by_metric('degree', top_n)
        
        # 
        weighted_dict = {author: score for author, score in weighted_top}
        unweighted_dict = {author: score for author, score in unweighted_top}
        
        # 
        weighted_authors = set(weighted_dict.keys())
        unweighted_authors = set(unweighted_dict.keys())
        
        common_authors = weighted_authors & unweighted_authors
        only_weighted = weighted_authors - unweighted_authors
        only_unweighted = unweighted_authors - weighted_authors
        
        print(f"\n :")
        print(f"  - : {len(common_authors)}")
        print(f"  - : {len(only_weighted)}")
        print(f"  - : {len(only_unweighted)}")
        
        # 
        rank_changes = {}
        weighted_rank = {author: i+1 for i, (author, _) in enumerate(weighted_top)}
        unweighted_rank = {author: i+1 for i, (author, _) in enumerate(unweighted_top)}
        
        for author in common_authors:
            rank_change = unweighted_rank[author] - weighted_rank[author]
            rank_changes[author] = rank_change
        
        # 
        if rank_changes:
            biggest_risers = sorted(rank_changes.items(), key=lambda x: x[1], reverse=True)[:5]
            biggest_fallers = sorted(rank_changes.items(), key=lambda x: x[1])[:5]
            
            print(f"\n  ( vs ):")
            for author, change in biggest_risers:
                if change > 0:
                    weighted_pos = weighted_rank[author]
                    unweighted_pos = unweighted_rank[author]
                    print(f"  {author}: {unweighted_pos}  {weighted_pos} (+{change})")
            
            print(f"\n  ( vs ):")
            for author, change in biggest_fallers:
                if change < 0:
                    weighted_pos = weighted_rank[author]
                    unweighted_pos = unweighted_rank[author]
                    print(f"  {author}: {unweighted_pos}  {weighted_pos} ({change})")
        
        return {
            'weighted_top': weighted_top,
            'unweighted_top': unweighted_top,
            'common_authors': common_authors,
            'only_weighted': only_weighted,
            'only_unweighted': only_unweighted,
            'rank_changes': rank_changes
        }
    
    def plot_weighted_comparison(self, top_n: int = 15, save_path: str = None):
        """
        
        
        Args:
            top_n: N
            save_path: 
        """
        print(f" ...")
        
        # N
        weighted_scores = {}
        unweighted_scores = {}
        
        for node in self.network.nodes():
            weighted_degree = sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True))
            weighted_scores[node] = weighted_degree
            unweighted_scores[node] = self.network.degree(node)
        
        # N
        top_weighted = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        authors = [author for author, _ in top_weighted]
        
        # 
        weighted_values = [weighted_scores[author] for author in authors]
        unweighted_values = [unweighted_scores[author] for author in authors]
        
        # 
        author_labels = []
        for author in authors:
            author_labels.append(author)
        
        # 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 
        bars1 = ax1.bar(range(len(authors)), weighted_values, color='skyblue', alpha=0.8)
        ax1.set_xlabel('Author Ranking', fontweight='bold', fontsize=FONT_CONFIG['label'])
        ax1.set_ylabel('Weighted Degree', fontweight='bold', fontsize=FONT_CONFIG['label'])
        ax1.set_title('a. Top 15 Authors - Weighted Degree Ranking', fontweight='bold', fontsize=FONT_CONFIG['title'])
        ax1.set_xticks(range(len(authors)))
        ax1.set_xticklabels(author_labels, rotation=45, ha='right', fontsize=FONT_CONFIG['tick'])
        ax1.grid(True, alpha=0.3)
        
        # 
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 
        bars2 = ax2.bar(range(len(authors)), unweighted_values, color='lightcoral', alpha=0.8)
        ax2.set_xlabel('Author Ranking', fontweight='bold', fontsize=FONT_CONFIG['label'])
        ax2.set_ylabel('Unweighted Degree', fontweight='bold', fontsize=FONT_CONFIG['label'])
        ax2.set_title('b. Same Authors - Unweighted Degree', fontweight='bold', fontsize=FONT_CONFIG['title'])
        ax2.set_xticks(range(len(authors)))
        ax2.set_xticklabels(author_labels, rotation=45, ha='right', fontsize=FONT_CONFIG['tick'])
        ax2.grid(True, alpha=0.3)
        
        # 
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" : {save_path}")
        
        plt.show()
    
    def visualize_with_cosmograph(self, network: nx.Graph = None):
        """
        cosmograph
        
        Args:
            network: None
        """
        try:
            import pandas as pd
            from cosmograph import cosmo
        except ImportError:
            print(" cosmograph: pip install cosmograph")
            return None
        
        if network is None:
            if self.filtered_network is None:
                print(" ")
                return None
            network = self.filtered_network
        
        print(f" cosmograph: {network.number_of_nodes()} , {network.number_of_edges()} ")
        
        # 
        node_data = []
        for i, node in enumerate(network.nodes()):
            node_info = network.nodes[node]
            
            # 
            weighted_degree = sum(data.get('weight', 1) for _, _, data in network.edges(node, data=True))
            
            # 
            label = node  # 
            
            # 
            paper_count = node_info.get('paper_count', 0)
            if paper_count >= 10:
                category = 'High'
            elif paper_count >= 5:
                category = 'Medium'
            else:
                category = 'Low'
            
            node_data.append({
                'id': i,
                'original_name': node,
                'label': label,
                'weighted_degree': weighted_degree,
                'paper_count': paper_count,
                'category': category
            })
        
        # 
        name_to_id = {node_data[i]['original_name']: i for i in range(len(node_data))}
        
        # 
        edge_data = []
        for u, v, data in network.edges(data=True):
            edge_data.append({
                'source': name_to_id[u],
                'target': name_to_id[v],
                'weight': data.get('weight', 1),
                'count': data.get('count', 1)
            })
        
        # DataFrame
        points_df = pd.DataFrame(node_data)
        links_df = pd.DataFrame(edge_data)
        
        print(f" : {len(points_df)} ")
        print(f" : {len(links_df)} ")
        
        # cosmograph
        widget = cosmo(
            points=points_df,
            links=links_df,
            point_id_by='id',
            link_source_by='source',
            link_target_by='target',
            point_color_by='category',
            point_size_by='weighted_degree',
            point_label_by='label',
            link_width_by='weight',
            point_include_columns=['weighted_degree', 'paper_count', 'original_name'],
            link_include_columns=['weight', 'count'],
        )
        
        print(" cosmograph")
        return widget
    
    def export_echarts_network_data(self, network: nx.Graph = None, communities_result: Dict = None, 
                                   output_file: str = "network_data.json", max_nodes: int = 500,
                                   target_communities: List[int] = None,
                                   node_size_scale: float = 10.0,
                                   node_size_min: int = 18,
                                   node_size_max: int = 160,
                                   label_font_min: int = 12,
                                   label_font_max: int = 24,
                                   exclude_non_largest: bool = True) -> str:
        """
        Export network data in ECharts format for web visualization
        
        Args:
            network: Network to visualize (default: filtered_network)
            communities_result: Community detection results for coloring
            output_file: JSON output filename
            max_nodes: Maximum nodes to include (for performance)
            
        Returns:
            str: Path to the generated JSON file
        """
        if network is None:
            if self.filtered_network is None:
                print(" Please filter network first or provide a network")
                return None
            network = self.filtered_network
            
        print(f" Preparing ECharts network data: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
        
        # Sample nodes if too many
        nodes_list = list(network.nodes())
        if len(nodes_list) > max_nodes:
            # Prioritize high-degree nodes
            node_degrees = [(node, sum(data.get('weight', 1) for _, _, data in network.edges(node, data=True))) 
                          for node in nodes_list]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, _ in node_degrees[:max_nodes]]
            network = network.subgraph(selected_nodes).copy()
            print(f"  Sampled to {len(selected_nodes)} high-degree nodes")
            
            #  
            isolated_after_sampling = [n for n in network.nodes() if network.degree(n) == 0]
            if isolated_after_sampling:
                print(f"   Found {len(isolated_after_sampling)} isolated nodes after sampling")
                print(f"   Removing isolated nodes for better visualization...")
                network.remove_nodes_from(isolated_after_sampling)
                print(f"   Final network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
        
        # 
        lcc_nodes = None
        if exclude_non_largest and network.number_of_nodes() > 0:
            try:
                lcc_nodes = max(nx.connected_components(network), key=len)
            except Exception:
                lcc_nodes = None

        # Prepare nodes data
        nodes_data = []
        categories_data = []
        category_names = set()
        
        # 
        allowed_partition = None
        partition_map = None
        if communities_result:
            partition_map = communities_result.get('partition', {})
        if communities_result and target_communities is not None:
            allowed_partition = set(target_communities)
        
        node_index = 0
        for node in network.nodes():
            node_data = network.nodes[node]
            
            # 
            if lcc_nodes is not None and node not in lcc_nodes:
                continue

            # Calculate node metrics
            weighted_degree = sum(data.get('weight', 1) for _, _, data in network.edges(node, data=True))
            paper_count = node_data.get('paper_count', 1)
            
            # Determine category (community-based only)
            if communities_result:
                # 
                if node not in partition_map:
                    continue
                category_id = partition_map[node]
                # 
                if allowed_partition is not None and category_id not in allowed_partition:
                    continue
                category = f"Community {category_id}"
            else:
                #  
                category = "Authors"
                category_id = 0
            
            category_names.add(category)
            
            # Clean node name for display - 
            if ',' in node:
                #  "Batty, Michael" -> "Batty"
                display_name = node.split(',')[0].strip()
            elif ' ' in node:
                #  "Michael Batty" -> "Batty"
                name_parts = node.split()
                display_name = name_parts[-1] if len(name_parts) > 1 else name_parts[0]
            else:
                # 
                display_name = node
            
            # 
            symbol_size = int(min(node_size_max, max(node_size_min, weighted_degree * node_size_scale)))
            font_size = int(min(label_font_max, max(label_font_min, weighted_degree * (node_size_scale/4))))

            node_info = {
                'id': str(node_index),
                'name': display_name,
                'full_name': node,
                'category': category_id,
                'value': weighted_degree,
                'symbolSize': symbol_size,  #  weighted degree
                'papers': paper_count,
                'weighted_degree': round(weighted_degree, 3),
                'label': {
                    'show': True,  #  
                    'fontSize': font_size,  #  
                    'position': 'right'
                }
            }
            
            nodes_data.append(node_info)
            node_index += 1
        
        # Create node name to ID mapping
        name_to_id = {node_info['full_name']: node_info['id'] for node_info in nodes_data}
        
        # Prepare links data
        links_data = []
        for u, v, data in network.edges(data=True):
            if u in name_to_id and v in name_to_id:
                weight = data.get('weight', 1)
                links_data.append({
                    'source': name_to_id[u],
                    'target': name_to_id[v],
                    'value': round(weight, 3),
                    'lineStyle': {
                        'width': min(10, max(2, weight * 4)),
                        'opacity': min(0.9, max(0.35, weight))
                    }
                })
        
        # Prepare categories data
        for i, category in enumerate(sorted(category_names)):
            categories_data.append({
                'name': category
            })
        
        # Create final data structure
        echarts_data = {
            'nodes': nodes_data,
            'links': links_data,
            'categories': categories_data,
            'network_stats': {
                'total_nodes': len(nodes_data),
                'total_links': len(links_data),
                'total_communities': len(categories_data),
                'max_weighted_degree': max(node['weighted_degree'] for node in nodes_data),
                'generation_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Save to JSON file
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(echarts_data, f, ensure_ascii=False, indent=2)
        
        print(f" ECharts network data exported to: {output_file}")
        print(f"   - Nodes: {len(nodes_data)}")
        print(f"   - Links: {len(links_data)}")
        print(f"   - Categories: {len(categories_data)}")
        
        return output_file
    
    def generate_echarts_html(self, data_file: str = "network_data.json", 
                            output_html: str = "network_visualization.html",
                            title: str = "EPB Author Collaboration Network",
                            force_show_all_labels: bool = True,
                            pixel_ratio: int = 3) -> str:
        """
        Generate HTML file with ECharts network visualization
        
        Args:
            data_file: JSON data file path
            output_html: Output HTML file path
            title: Chart title
            
        Returns:
            str: Path to generated HTML file
        """
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EPB Network</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #ffffff;
            width: 100vw;
            height: 100vh;
        }}
        .container {{
            width: 100vw;
            height: 100vh;
            margin: 0;
            padding: 0;
        }}
        .title {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 24px;
            font-weight: bold;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-number {{
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 5px;
        }}
        #network-chart {{
            width: 100vw;
            height: 92vh;
            background: white;
        }}
        .controls {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
            font-size: 14px;
            transition: background 0.3s;
        }}
        .btn:hover {{
            background: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!--   -->
        
        <div class="stats" id="network-stats">
            <!-- Stats will be populated by JavaScript -->
        </div>
        
        <div class="controls">
            <button class="btn" onclick="resetZoom()"> Reset Zoom</button>
            <button class="btn" onclick="saveAsImage()"> Save as PNG</button>
            <button class="btn" onclick="toggleLabels()"> Toggle Labels</button>
            <button class="btn" onclick="toggleLegend()" id="legend-btn"> Show Legend</button>
            <button class="btn" onclick="toggleLayout()" id="layout-btn"> Lock Layout</button>
            <span style="margin-left:15px;color:#555;font-weight:600;">Node</span>
            <input id="nodeScale" type="range" min="0.5" max="3.0" step="0.1" value="1.0" 
                   oninput="updateSizes()" style="vertical-align:middle;width:120px;">
            <span id="nodeScaleValue" style="color:#3498db;font-weight:600;width:35px;display:inline-block;">1.0</span>
            <span style="margin-left:15px;color:#555;font-weight:600;">Label</span>
            <input id="labelScale" type="range" min="0.5" max="3.5" step="0.1" value="1.0" 
                   oninput="updateSizes()" style="vertical-align:middle;width:120px;">
            <span id="labelScaleValue" style="color:#3498db;font-weight:600;width:35px;display:inline-block;">1.0</span>
        </div>
        
        <div id="network-chart"></div>
    </div>

    <script>
        // Initialize ECharts
        var myChart = echarts.init(document.getElementById('network-chart'));
        var showLabels = true;
        var showLegend = false;  //  
        var layoutLocked = false;  //  
        var savedPositions = null;  //  
        var originalNodes = null;  // 
        var nodeScaleFactor = 1.0;
        var labelScaleFactor = 1.0;
        
        // Show loading
        myChart.showLoading({{
            text: 'Loading network data...',
            color: '#3498db',
            textColor: '#2c3e50',
            maskColor: 'rgba(255, 255, 255, 0.8)',
        }});

        // Load network data
        $.getJSON('{data_file}', function (networkData) {{
            myChart.hideLoading();
            
            // 
            originalNodes = JSON.parse(JSON.stringify(networkData.nodes));
            
            // Update stats
            updateStats(networkData.network_stats);
            
            // Configure chart options
            var option = {{
                //  
                // title: {{
                //     text: '{title}',
                //     left: 'center',
                //     textStyle: {{
                //         color: '#2c3e50',
                //         fontSize: 20,
                //         fontWeight: 'bold'
                //     }}
                // }},
                tooltip: {{
                    trigger: 'item',
                    formatter: function(params) {{
                        if (params.dataType === 'node') {{
                            return `
                                <strong>${{params.data.full_name}}</strong><br/>
                                Papers: ${{params.data.papers}}<br/>
                                Weighted Degree: ${{params.data.weighted_degree}}<br/>
                                Category: ${{networkData.categories[params.data.category].name}}
                            `;
                        }} else if (params.dataType === 'edge') {{
                            return `
                                <strong>Collaboration</strong><br/>
                                Weight: ${{params.data.value}}<br/>
                                Between: ${{params.data.source}} - ${{params.data.target}}
                            `;
                        }}
                    }},
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderColor: '#3498db',
                    borderWidth: 2,
                    textStyle: {{
                        color: '#2c3e50',
                        fontSize: 12
                    }}
                }},
                legend: [{{
                    data: networkData.categories.map(function (a) {{
                        return a.name;
                    }}),
                    top: 'bottom',
                    show: false,  //  
                    textStyle: {{
                        color: '#2c3e50'
                    }}
                }}],
                series: [{{
                    name: 'EPB Network',
                    type: 'graph',
                    layout: 'force',
                    data: networkData.nodes,
                    links: networkData.links,
                    categories: networkData.categories,
                    roam: true,
                    draggable: true,  //  
                    focusNodeAdjacency: true,
                    itemStyle: {{
                        borderColor: '#fff',
                        borderWidth: 1,
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.3)'
                    }},
                    label: {{
                        show: {str(force_show_all_labels).lower()},
                        position: 'right',
                        formatter: '{{b}}',
                        fontSize: function(params) {{
                            // fontSize
                            return params.data.label ? params.data.label.fontSize : 12;
                        }},
                        fontWeight: 'bold',
                        color: '#2c3e50'
                    }},
                    labelLayout: {{
                        hideOverlap: false  //  
                    }},
                    scaleLimit: {{
                        min: 0.02,  //  
                        max: 10     //  
                    }},
                    force: {{
                        repulsion: 2000,      //  
                        edgeLength: [100, 400], //  
                        layoutAnimation: true,
                        friction: 0.6,        //  
                        gravity: 0.1          //  
                    }},
                    lineStyle: {{
                        color: 'source',
                        curveness: 0.1,
                        opacity: 0.6
                    }},
                    emphasis: {{
                        focus: 'adjacency',
                        lineStyle: {{
                            opacity: 0.9,
                            width: 4
                        }},
                        itemStyle: {{
                            shadowBlur: 20,
                            shadowColor: '#3498db'
                        }}
                    }}
                }}]
            }};

            myChart.setOption(option);
            
            // Handle window resize
            window.addEventListener('resize', function() {{
                myChart.resize();
            }});
        }})
        .fail(function() {{
            myChart.hideLoading();
            alert('Failed to load network data. Please check the data file path.');
        }});

        // Utility functions
        function updateStats(stats) {{
            const statsHtml = `
                <div class="stat-item">
                    <div class="stat-number">${{stats.total_nodes}}</div>
                    <div class="stat-label">Authors</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${{stats.total_links}}</div>
                    <div class="stat-label">Collaborations</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${{stats.total_communities}}</div>
                    <div class="stat-label">Communities</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${{stats.max_weighted_degree}}</div>
                    <div class="stat-label">Max Degree</div>
                </div>
            `;
            document.getElementById('network-stats').innerHTML = statsHtml;
        }}

        function resetZoom() {{
            myChart.dispatchAction({{
                type: 'restore'
            }});
            //  
            setTimeout(function() {{
                myChart.dispatchAction({{
                    type: 'graphRoam',
                    zoom: 0.3  // 30%
                }});
            }}, 100);
        }}

        function saveAsImage() {{
            const url = myChart.getDataURL({{
                type: 'png',
                pixelRatio: {pixel_ratio},
                backgroundColor: '#fff'
            }});
            
            const link = document.createElement('a');
            link.download = 'epb-network.png';
            link.href = url;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        function toggleLabels() {{
            showLabels = !showLabels;
            myChart.setOption({{
                series: [{{
                    label: {{
                        show: showLabels
                    }}
                }}]
            }});
        }}

        function toggleLegend() {{
            //  /
            showLegend = !showLegend;
            const legendBtn = document.getElementById('legend-btn');
            
            myChart.setOption({{
                legend: {{
                    show: showLegend
                }}
            }});
            
            // 
            if (showLegend) {{
                legendBtn.innerHTML = ' Hide Legend';
                legendBtn.style.background = '#e74c3c';
            }} else {{
                legendBtn.innerHTML = ' Show Legend';
                legendBtn.style.background = '#3498db';
            }}
        }}

        function toggleLayout() {{
            //  /
            layoutLocked = !layoutLocked;
            const layoutBtn = document.getElementById('layout-btn');
            
            if (layoutLocked) {{
                // 
                myChart.setOption({{
                    series: [{{
                        layout: 'none',  // 
                        force: {{
                            layoutAnimation: false  // 
                        }}
                    }}]
                }}, false);  //  merge
                
                layoutBtn.innerHTML = ' Unlock Layout';
                layoutBtn.style.background = 'linear-gradient(135deg, #f39c12 0%, #e67e22 100%)';
                console.log(' Layout locked - You can now manually drag nodes to optimize space');
            }} else {{
                // 
                myChart.setOption({{
                    series: [{{
                        layout: 'force',  // 
                        force: {{
                            repulsion: 800,
                            edgeLength: [50, 150],
                            gravity: 0.1,
                            center: ['50%', '50%'],
                            friction: 0.9,
                            layoutAnimation: true,
                            preventOverlap: false
                        }}
                    }}]
                }}, false);  //  merge
                
                layoutBtn.innerHTML = ' Lock Layout';
                layoutBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                console.log(' Layout unlocked - Force-directed algorithm resumed');
            }}
        }}

        function updateSizes() {{
            // 
            nodeScaleFactor = parseFloat(document.getElementById('nodeScale').value);
            labelScaleFactor = parseFloat(document.getElementById('labelScale').value);
            
            // 
            document.getElementById('nodeScaleValue').innerText = nodeScaleFactor.toFixed(1);
            document.getElementById('labelScaleValue').innerText = labelScaleFactor.toFixed(1);
            
            // 
            const scaledNodes = originalNodes.map(node => {{
                const scaled = JSON.parse(JSON.stringify(node));
                scaled.symbolSize = Math.round(node.symbolSize * nodeScaleFactor);
                if (scaled.label) {{
                    scaled.label.fontSize = Math.round(node.label.fontSize * labelScaleFactor);
                }}
                return scaled;
            }});
            
            // 
            myChart.setOption({{
                series: [{{
                    data: scaledNodes
                }}]
            }}, {{notMerge: false, replaceMerge: ['series[0].data']}});
        }}
    </script>
</body>
</html>
        """
        
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f" ECharts HTML visualization generated: {output_html}")
        print(f"   Features:")
        print(f"   - Interactive network with force layout")
        print(f"   - Community-based coloring")
        print(f"   - Node size based on weighted degree")
        print(f"   - Hover tooltips with detailed info")
        print(f"   - PNG export capability")
        print(f"   - Zoom, pan, and label toggle")
        
        return output_html
    
    def create_echarts_visualization(self, communities_result: Dict = None, max_nodes: int = 500,
                                   output_dir: str = "Figures",
                                   target_communities: List[int] = None,
                                   top_k_communities: int = None,
                                   by: str = 'size',
                                   node_size_scale: float = 10.0,
                                   node_size_min: int = 18,
                                   node_size_max: int = 160,
                                   label_font_min: int = 12,
                                   label_font_max: int = 24) -> Tuple[str, str]:
        """
        Complete workflow: export data and generate ECharts HTML visualization
        
        Args:
            communities_result: Community detection results
            max_nodes: Maximum nodes to include
            output_dir: Output directory for files
            
        Returns:
            Tuple[str, str]: Paths to (JSON data file, HTML file)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        data_file = os.path.join(output_dir, "network_data.json")
        html_file = os.path.join(output_dir, "01-network-visualization.html")
        
        print(" Creating ECharts network visualization...")
        
        # Derive top-K communities if requested
        selected_communities = None
        if communities_result is not None:
            partition = communities_result.get('partition', {})
            sizes = communities_result.get('community_sizes', {})
            if top_k_communities is not None and top_k_communities > 0 and sizes:
                # sort by size desc
                sorted_ids = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
                selected_communities = [cid for cid, _ in sorted_ids[:top_k_communities]]
            if target_communities is not None:
                selected_communities = target_communities

        # Export data with community filter if any
        self.export_echarts_network_data(
            communities_result=communities_result,
            output_file=data_file,
            max_nodes=max_nodes,
            target_communities=selected_communities,
            node_size_scale=node_size_scale,
            node_size_min=node_size_min,
            node_size_max=node_size_max,
            label_font_min=label_font_min,
            label_font_max=label_font_max
        )
        
        # Generate HTML
        self.generate_echarts_html(
            data_file=os.path.basename(data_file),  # Use relative path in HTML
            output_html=html_file,
            force_show_all_labels=True,
            pixel_ratio=3
        )
        
        print(f" ECharts visualization complete!")
        print(f"    Data: {data_file}")
        print(f"    HTML: {html_file}")
        print(f"    Open {html_file} in browser to view")
        
        return data_file, html_file

class NetworkStructuralAnalyzer:
    """"""
    
    def __init__(self, network: nx.Graph):
        """
        
        
        Args:
            network: 
        """
        self.network = network
        self.largest_cc = None
        self.largest_subgraph = None
        self._prepare_largest_component()
    
    def _prepare_largest_component(self):
        """"""
        if self.network.number_of_nodes() == 0:
            return
        
        # 
        connected_components = list(nx.connected_components(self.network))
        if connected_components:
            self.largest_cc = max(connected_components, key=len)
            self.largest_subgraph = self.network.subgraph(self.largest_cc).copy()
            print(f" Largest connected component: {len(self.largest_cc)} nodes")
    
    def analyze_degree_distribution(self, use_weighted: bool = True, plot: bool = True):
        """
        1: 
        
        Args:
            use_weighted: 
            plot: 
        """
        print("\n" + "="*50)
        print(" 1: ")
        print("="*50)
        
        # 
        if use_weighted:
            degrees = []
            for node in self.network.nodes():
                weighted_degree = sum(data.get('weight', 1) for _, _, data in self.network.edges(node, data=True))
                degrees.append(weighted_degree)
            degree_type = "Weighted Degree"
        else:
            degrees = [d for n, d in self.network.degree()]
            degree_type = "Unweighted Degree"
        
        degrees = np.array(degrees)
        
        # 
        stats = {
            'min': np.min(degrees),
            'max': np.max(degrees),
            'mean': np.mean(degrees),
            'median': np.median(degrees),
            'std': np.std(degrees)
        }
        
        degree_type_en = degree_type
        print(f"{degree_type_en} Distribution Statistics:")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}")
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle('Degree Distribution Analysis', fontsize=20, fontweight='bold', y=0.98)
            
            # Degree distribution histogram with professional styling
            n, bins, patches = ax1.hist(degrees, bins=30, alpha=0.8, 
                                       color=JOURNAL_COLORS['primary'], 
                                       edgecolor='white', linewidth=1.2)
            
            # Add gradient effect to histogram
            for i, p in enumerate(patches):
                alpha = 0.3 + 0.7 * (n[i] / max(n))
                p.set_alpha(alpha)
            
            ax1.set_xlabel(f'{degree_type}', 
                          fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title('a. Degree Distribution', fontweight='bold', loc='left', fontsize=FONT_CONFIG['title'])
            
            # Add statistics box
            textstr = f'Mean: {stats["mean"]:.2f}\nStd: {stats["std"]:.2f}\nMax: {stats["max"]:.2f}'
            props = dict(boxstyle='round', facecolor=JOURNAL_COLORS['light'], alpha=0.8)
            ax1.text(0.75, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
            
            # Power-law examination: log-log plot with enhanced styling
            degree_counts = pd.Series(degrees).value_counts().sort_index()
            x = degree_counts.index.values
            y = degree_counts.values
            
            # Filter zero values for log transformation
            mask = (x > 0) & (y > 0)
            x_filtered = x[mask]
            y_filtered = y[mask]
            
            if len(x_filtered) > 0:
                ax2.loglog(x_filtered, y_filtered, 'o', color=JOURNAL_COLORS['secondary'], 
                          markersize=8, alpha=0.8, markeredgecolor='white', markeredgewidth=1)
                
                # Add power-law fit line
                if len(x_filtered) > 3:
                    log_x = np.log10(x_filtered)
                    log_y = np.log10(y_filtered)
                    z = np.polyfit(log_x, log_y, 1)
                    fit_line = 10**(z[1]) * x_filtered**z[0]
                    ax2.loglog(x_filtered, fit_line, '--', 
                              color=JOURNAL_COLORS['accent'], linewidth=2.5, alpha=0.9,
                              label=f'Power law fit ( = {-z[0]:.2f})')
                    ax2.legend(frameon=True, fancybox=True, shadow=True)
                
                ax2.set_xlabel(f'log({degree_type})', 
                              fontweight='bold')
                ax2.set_ylabel('log(Frequency)', fontweight='bold')
                ax2.set_title('b. Power-law Test (log-log scale)', fontweight='bold', loc='left', fontsize=FONT_CONFIG['title'])
            
            plt.tight_layout()
            plt.show()
        
        return stats
    
    def create_combined_structural_analysis_plots(self, filtered_network=None, save_path: str = None):
        """
        23
        -
        : a(-), b(-), c(-),
                 d(-), e(-), f(-)
        
        Args:
            filtered_network: Noneself.network
            save_path: 
        """
        print("\n" + "="*60)
        print(" Creating Combined Structural Analysis (23 Layout)")
        print("="*60)
        
        # 23
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        # 
        # fig.suptitle('Structural Analysis: Degree Correlation and Clustering Coefficient', 
        #              fontsize=16, fontweight='bold', y=0.95)
        
        # 
        networks = [
            ('full', 'Full Network', self.network),
            ('filtered', 'Filtered Network', filtered_network if filtered_network is not None else self.network),
            ('largest_component', 'Largest Component', self.largest_subgraph)
        ]
        
        # 
        results = {}
        
        # - (a, b, c)
        for idx, (network_type, network_name, analysis_network) in enumerate(networks):
            if analysis_network is None or analysis_network.number_of_nodes() == 0:
                print(f" Cannot analyze {network_name}: Network is empty or not connected")
                continue
                
            ax = axes[0, idx]  # 
            
            # -
            weighted_degrees = {}
            for node in analysis_network.nodes():
                weighted_degree = sum(data.get('weight', 1) for _, _, data in analysis_network.edges(node, data=True))
                weighted_degrees[node] = weighted_degree
            
            avg_neighbor_weighted_degree = {}
            for node in analysis_network.nodes():
                neighbors = list(analysis_network.neighbors(node))
                if neighbors:
                    avg_neighbor_weighted_degree[node] = np.mean([weighted_degrees[neighbor] for neighbor in neighbors])
                else:
                    avg_neighbor_weighted_degree[node] = 0
            
            degrees = np.array([weighted_degrees[node] for node in analysis_network.nodes()])
            neighbor_degrees = np.array([avg_neighbor_weighted_degree[node] for node in analysis_network.nodes()])
            
            # 
            correlation = np.corrcoef(degrees, neighbor_degrees)[0, 1]
            
            # 
            scatter = ax.scatter(degrees, neighbor_degrees, 
                               c=degrees, cmap='plasma', 
                               s=20, alpha=0.7, 
                               edgecolors='white', linewidth=0.3,
                               vmin=np.percentile(degrees, 5),
                               vmax=np.percentile(degrees, 95))
            
            # 
            if len(degrees) > 1:
                z = np.polyfit(degrees, neighbor_degrees, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(degrees.min(), degrees.max(), 100)
                y_trend = p(x_trend)
                
                ax.plot(x_trend, y_trend, "--", 
                       color='red', linewidth=2, alpha=0.8,
                       label=f'r = {correlation:.3f}')
            
            # 
            ax.set_xlabel('Node Degree', fontweight='bold', fontsize=9)
            ax.set_ylabel('Avg Neighbor Degree', fontweight='bold', fontsize=9)
            ax.set_title(f'{chr(97+idx)}. {network_name}', fontweight='bold', fontsize=FONT_CONFIG['title'])  # a, b, c
            
            # 
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=8)
            
            # 
            if network_type not in results:
                results[network_type] = {}
            results[network_type]['correlation'] = {
                'correlation': correlation,
                'degrees': degrees,
                'neighbor_degrees': neighbor_degrees
            }
        
        #  (d, e, f)
        for idx, (network_type, network_name, analysis_network) in enumerate(networks):
            if analysis_network is None or analysis_network.number_of_nodes() == 0:
                continue
                
            ax = axes[1, idx]  # 
            
            # 
            clustering = nx.clustering(analysis_network, weight='weight')
            avg_clustering = nx.average_clustering(analysis_network, weight='weight')
            
            # 
            weighted_degrees = {}
            for node in analysis_network.nodes():
                weighted_degree = sum(data.get('weight', 1) for _, _, data in analysis_network.edges(node, data=True))
                weighted_degrees[node] = weighted_degree
            
            # 
            degree_clustering = {}
            for node in analysis_network.nodes():
                degree = int(round(weighted_degrees[node]))
                if degree not in degree_clustering:
                    degree_clustering[degree] = []
                degree_clustering[degree].append(clustering[node])
            
            # 
            degrees = sorted(degree_clustering.keys())
            avg_clustering_by_degree = []
            for degree in degrees:
                avg_clustering_by_degree.append(np.mean(degree_clustering[degree]))
            
            if len(degrees) <= 1:
                continue
            
            # 
            raw_sizes = [len(degree_clustering[d]) for d in degrees]
            max_raw_size = max(raw_sizes)
            sizes = [40 + (size/max_raw_size) * 160 for size in raw_sizes]  # 
            
            scatter = ax.scatter(degrees, avg_clustering_by_degree, 
                               c=avg_clustering_by_degree, cmap='plasma',
                               s=sizes, alpha=0.7, 
                               edgecolors='white', linewidth=0.8,
                               vmin=0, vmax=max(avg_clustering_by_degree))
            
            # 
            if len(degrees) > 2:
                z = np.polyfit(degrees, avg_clustering_by_degree, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(degrees), max(degrees), 100)
                y_trend = p(x_trend)
                ax.plot(x_trend, y_trend, "--", 
                       color='green', linewidth=2, alpha=0.8,
                       label=f'Slope = {z[0]:.4f}')
            
            # 
            ax.axhline(y=avg_clustering, color='orange', 
                      linestyle='-', alpha=0.7, linewidth=2,
                      label=f'Global avg = {avg_clustering:.3f}')
            
            # 
            ax.set_xlabel('Node Degree', fontweight='bold', fontsize=FONT_CONFIG['label'])
            ax.set_ylabel('Avg Clustering Coeff', fontweight='bold', fontsize=FONT_CONFIG['label'])
            # 
            # ax.set_title(f'({chr(100+idx)}) {network_name}', fontweight='bold', fontsize=10)  # d, e, f
            
            # 
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', fontsize=8)
            
            # y
            ax.set_ylim(0, max(avg_clustering_by_degree) * 1.1)
            
            # 
            if network_type not in results:
                results[network_type] = {}
            results[network_type]['clustering'] = {
                'average_clustering': avg_clustering,
                'clustering_by_degree': dict(zip(degrees, avg_clustering_by_degree))
            }
        
        # 
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # 
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f" Combined structural analysis plot saved to: {save_path}")
        
        plt.show()
        return results

    def analyze_degree_correlation(self, plot: bool = True, save_path: str = None, use_network: str = 'largest_component'):
        """
        Step 2: Degree-degree correlation analysis for assortative/disassortative networks
        
        Args:
            plot: Whether to create visualizations
            save_path: Path to save the plot
            use_network: Which network to use ('full', 'filtered', 'largest_component')
        """
        print("\n" + "="*50)
        print(" Step 2: Degree-Degree Correlation Analysis")
        print("="*50)
        
        #  
        if use_network == 'full':
            analysis_network = self.network
            network_name = ""
        elif use_network == 'filtered':
            analysis_network = self.network  # network
            network_name = ""
        elif use_network == 'largest_component':
            analysis_network = self.largest_subgraph
            network_name = ""
        else:
            raise ValueError(f"Invalid use_network parameter: {use_network}")
        
        if analysis_network is None or analysis_network.number_of_nodes() == 0:
            print(f" Cannot analyze {network_name}: Network is empty or not connected")
            return None
        
        print(f" {network_name}: {analysis_network.number_of_nodes()} , {analysis_network.number_of_edges()} ")
        
        # Calculate weighted degrees for each node
        weighted_degrees = {}
        for node in analysis_network.nodes():
            weighted_degree = sum(data.get('weight', 1) for _, _, data in analysis_network.edges(node, data=True))
            weighted_degrees[node] = weighted_degree
        
        # Calculate average neighbor weighted degree
        avg_neighbor_weighted_degree = {}
        for node in analysis_network.nodes():
            neighbors = list(analysis_network.neighbors(node))
            if neighbors:
                avg_neighbor_weighted_degree[node] = np.mean([weighted_degrees[neighbor] for neighbor in neighbors])
            else:
                avg_neighbor_weighted_degree[node] = 0
        
        # Prepare data for analysis
        degrees = []
        neighbor_degrees = []
        for node in analysis_network.nodes():
            degrees.append(weighted_degrees[node])
            neighbor_degrees.append(avg_neighbor_weighted_degree[node])
        
        degrees = np.array(degrees)
        neighbor_degrees = np.array(neighbor_degrees)
        
        # 
        correlation = np.corrcoef(degrees, neighbor_degrees)[0, 1]
        
        print(f"Degree-degree correlation coefficient: {correlation:.4f}")
        if correlation > 0.1:
            network_type = "Assortative Network (high-degree nodes tend to connect to high-degree nodes)"
        elif correlation < -0.1:
            network_type = "Disassortative Network (high-degree nodes tend to connect to low-degree nodes)"
        else:
            network_type = "Neutral Network (no significant degree-degree correlation)"
        
        print(f"Network type: {network_type}")
        
        if plot:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot with sophisticated color mapping
            # Use a beautiful color scheme that highlights patterns better
            scatter = ax.scatter(degrees, neighbor_degrees, 
                               c=degrees, cmap='plasma', 
                               s=50, alpha=0.75, 
                               edgecolors='white', linewidth=0.6,
                               vmin=np.percentile(degrees, 5),
                               vmax=np.percentile(degrees, 95))
            
            # Add trend line with confidence interval
            z = np.polyfit(degrees, neighbor_degrees, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(degrees.min(), degrees.max(), 100)
            y_trend = p(x_trend)
            
            ax.plot(x_trend, y_trend, "--", 
                   color=JOURNAL_COLORS['accent'], linewidth=3, alpha=0.9,
                   label=f'Linear fit (r = {correlation:.3f})')
            
            # Add confidence interval
            residuals = neighbor_degrees - p(degrees)
            std_error = np.std(residuals)
            ax.fill_between(x_trend, y_trend - 1.96*std_error, y_trend + 1.96*std_error,
                           alpha=0.2, color=JOURNAL_COLORS['accent'])
            
            # Styling
            ax.set_xlabel('Node Degree', fontweight='bold')
            ax.set_ylabel('Average Neighbor Degree', fontweight='bold')
            
            # Extract network type for display
            if "Assortative" in network_type:
                network_type_short = "Assortative"
            elif "Disassortative" in network_type:
                network_type_short = "Disassortative"
            else:
                network_type_short = "Neutral"
            ax.set_title(f'Correlation: {correlation:.4f} ({network_type_short} Network)', 
                        fontweight='bold', fontsize=FONT_CONFIG['title'])
            
            # Add sophisticated colorbar with better styling
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=25, pad=0.02)
            cbar.set_label('Weighted Degree', fontweight='bold', fontsize=FONT_CONFIG['label'])
            cbar.ax.tick_params(labelsize=FONT_CONFIG['tick'])
            
            # Add gradient effect to colorbar
            cbar.outline.set_linewidth(1.5)
            cbar.outline.set_edgecolor('gray')
            
            # Customize colorbar ticks
            cbar_ticks = np.linspace(np.percentile(degrees, 5), np.percentile(degrees, 95), 6)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([f'{tick:.1f}' for tick in cbar_ticks])
            
            # Add legend
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
            
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f" Degree correlation plot saved to: {save_path}")
            
            plt.show()
        
        return {
            'correlation': correlation,
            'network_type': network_type,
            'degrees': degrees,
            'neighbor_degrees': neighbor_degrees
        }
    
    def analyze_clustering_coefficient(self, plot: bool = True, save_path: str = None, use_network: str = 'largest_component'):
        """
        Step 3: Clustering coefficient analysis
        
        Args:
            plot: Whether to create visualizations
            save_path: Path to save the plot
            use_network: Which network to use ('full', 'filtered', 'largest_component')
        """
        print("\n" + "="*50)
        print(" Step 3: Clustering Coefficient Analysis")
        print("="*50)
        
        #  
        if use_network == 'full':
            analysis_network = self.network
            network_name = ""
        elif use_network == 'filtered':
            analysis_network = self.network  # network
            network_name = ""
        elif use_network == 'largest_component':
            analysis_network = self.largest_subgraph
            network_name = ""
        else:
            raise ValueError(f"Invalid use_network parameter: {use_network}")
        
        if analysis_network is None or analysis_network.number_of_nodes() == 0:
            print(f" Cannot analyze {network_name}: Network is empty or not connected")
            return None
        
        print(f" {network_name}: {analysis_network.number_of_nodes()} , {analysis_network.number_of_edges()} ")
        
        # Calculate weighted clustering coefficient
        clustering = nx.clustering(analysis_network, weight='weight')
        avg_clustering = nx.average_clustering(analysis_network, weight='weight')
        
        print(f"Average weighted clustering coefficient: {avg_clustering:.4f}")
        
        # Calculate weighted degrees for grouping
        weighted_degrees = {}
        for node in analysis_network.nodes():
            weighted_degree = sum(data.get('weight', 1) for _, _, data in analysis_network.edges(node, data=True))
            weighted_degrees[node] = weighted_degree
        
        # Group clustering coefficients by weighted degree
        degree_clustering = {}
        for node in analysis_network.nodes():
            degree = int(round(weighted_degrees[node]))  # Round weighted degree for grouping
            if degree not in degree_clustering:
                degree_clustering[degree] = []
            degree_clustering[degree].append(clustering[node])
        
        # 
        degrees = sorted(degree_clustering.keys())
        avg_clustering_by_degree = []
        
        for degree in degrees:
            avg_clustering_by_degree.append(np.mean(degree_clustering[degree]))
        
        if plot and len(degrees) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plot with well-balanced size based on number of nodes at each degree
            # Adjust size range for better visibility
            raw_sizes = [len(degree_clustering[d]) for d in degrees]
            max_raw_size = max(raw_sizes)
            # Scale sizes: minimum 80, maximum 400 for better visibility
            sizes = [80 + (size/max_raw_size) * 320 for size in raw_sizes]
            
            scatter = ax.scatter(degrees, avg_clustering_by_degree, 
                               c=avg_clustering_by_degree, cmap='plasma',
                               s=sizes, alpha=0.75, 
                               edgecolors='white', linewidth=1.2,
                               vmin=0, vmax=max(avg_clustering_by_degree))
            
            # Add trend line
            if len(degrees) > 2:
                z = np.polyfit(degrees, avg_clustering_by_degree, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(degrees), max(degrees), 100)
                y_trend = p(x_trend)
                ax.plot(x_trend, y_trend, "--", 
                       color=JOURNAL_COLORS['success'], linewidth=3, alpha=0.9,
                       label=f'Linear trend (slope = {z[0]:.4f})')
            
            # Add horizontal line for average clustering
            ax.axhline(y=avg_clustering, color=JOURNAL_COLORS['neutral'], 
                      linestyle='-', alpha=0.7, linewidth=2,
                      label=f'Global average = {avg_clustering:.4f}')
            
            # Styling
            ax.set_xlabel('Node Degree', fontweight='bold', fontsize=FONT_CONFIG['label'])
            ax.set_ylabel('Average Clustering Coefficient', fontweight='bold', fontsize=FONT_CONFIG['label'])
            ax.set_title('Clustering Coefficient Analysis', 
                        fontweight='bold', fontsize=FONT_CONFIG['title'])
            
            # Add sophisticated colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=25, pad=0.02)
            cbar.set_label('Average Clustering Coefficient', fontweight='bold', fontsize=FONT_CONFIG['label'])
            cbar.ax.tick_params(labelsize=FONT_CONFIG['tick'])
            
            # Customize colorbar appearance
            cbar.outline.set_linewidth(1.5)
            cbar.outline.set_edgecolor('gray')
            
            # Add legend
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
            
            # Set y-axis limits for better visualization
            ax.set_ylim(0, max(avg_clustering_by_degree) * 1.1)
            
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f" Clustering coefficient plot saved to: {save_path}")
            
            plt.show()
        
        return {
            'average_clustering': avg_clustering,
            'clustering_by_node': clustering,
            'clustering_by_degree': dict(zip(degrees, avg_clustering_by_degree))
        }
    
    def analyze_small_world(self, plot: bool = True, save_path: str = None):
        """
        Step 4: Small-world effect analysis
        
        Args:
            plot: Whether to create visualizations
            save_path: Path to save the plot
        """
        print("\n" + "="*50)
        print(" Step 4: Small-World Effect Analysis")
        print("="*50)
        
        if self.largest_subgraph is None:
            print(" Cannot analyze: Network is not connected")
            return None
        
        sample_subgraph = self.largest_subgraph
        
        try:
            # 
            path_lengths = dict(nx.all_pairs_shortest_path_length(sample_subgraph))
            
            # 
            all_distances = []
            for source in path_lengths:
                for target, distance in path_lengths[source].items():
                    if source != target:
                        all_distances.append(distance)
            
            if all_distances:
                diameter = max(all_distances)
                avg_path_length = np.mean(all_distances)
                
                print(f"Network diameter: {diameter}")
                print(f"Average path length: {avg_path_length:.2f}")
                
                # 
                max_distance = max(all_distances)
                coverage_by_distance = {}
                total_pairs = len(all_distances)
                
                for d in range(1, max_distance + 1):
                    coverage = sum(1 for dist in all_distances if dist <= d) / total_pairs
                    coverage_by_distance[d] = coverage
                    if coverage >= 0.9 and d not in [k for k, v in coverage_by_distance.items() if v >= 0.9][:1]:
                        print(f"90% coverage requires: {d} steps")
                
                if plot:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                    # fig.suptitle('Small-World Network Analysis', fontsize=20, fontweight='bold', y=0.98)
                    
                    # Path length distribution with enhanced styling
                    bins = range(1, max_distance+2)
                    n, bins_edges, patches = ax1.hist(all_distances, bins=bins, 
                                                     alpha=0.8, color=JOURNAL_COLORS['primary'],
                                                     edgecolor='white', linewidth=1)
                    
                    # Color gradient for histogram
                    for i, p in enumerate(patches):
                        p.set_facecolor(plt.cm.Blues(0.4 + 0.6 * n[i] / max(n)))
                    
                    ax1.axvline(x=avg_path_length, color=JOURNAL_COLORS['accent'], 
                               linestyle='--', linewidth=3, alpha=0.8,
                               label=f'Mean = {avg_path_length:.2f}')
                    ax1.axvline(x=diameter, color=JOURNAL_COLORS['success'], 
                               linestyle='-', linewidth=3, alpha=0.8,
                               label=f'Diameter = {diameter}')
                    
                    ax1.set_xlabel('Path Length', fontweight='bold', fontsize=FONT_CONFIG['label'])
                    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=FONT_CONFIG['label'])
                    ax1.set_title('a. Path Length Distribution', fontweight='bold', loc='center', fontsize=FONT_CONFIG['title'])
                    ax1.legend(frameon=True, fancybox=True, shadow=True)
                    
                    # Coverage curve with professional styling
                    distances = list(coverage_by_distance.keys())
                    coverages = list(coverage_by_distance.values())
                    
                    ax2.plot(distances, coverages, 'o-', 
                            color=JOURNAL_COLORS['secondary'], linewidth=3, 
                            markersize=8, markerfacecolor='white', 
                            markeredgecolor=JOURNAL_COLORS['secondary'], 
                            markeredgewidth=2, alpha=0.9)
                    
                    # Add 90% coverage reference line
                    ax2.axhline(y=0.9, color=JOURNAL_COLORS['success'], 
                               linestyle='--', linewidth=2, alpha=0.8,
                               label='90% Coverage')
                    
                    # Fill area under curve
                    ax2.fill_between(distances, coverages, alpha=0.3, 
                                    color=JOURNAL_COLORS['secondary'])
                    
                    ax2.set_xlabel('Number of Steps', fontweight='bold', fontsize=FONT_CONFIG['label'])
                    ax2.set_ylabel('Network Coverage Ratio', fontweight='bold', fontsize=FONT_CONFIG['label'])
                    ax2.set_title('b. Network Reachability Analysis', fontweight='bold', loc='center', fontsize=FONT_CONFIG['title'])
                    ax2.set_ylim(0, 1.05)
                    ax2.legend(frameon=True, fancybox=True, shadow=True)
                    
                    # Add grid with custom style
                    for ax in [ax1, ax2]:
                        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                        ax.set_axisbelow(True)
                    
                    plt.tight_layout()
                    
                    # Save figure if path provided
                    if save_path:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                        print(f" Small world analysis plot saved to: {save_path}")
                    
                    plt.show()
                
                return {
                    'diameter': diameter,
                    'average_path_length': avg_path_length,
                    'coverage_by_distance': coverage_by_distance,
                    'is_small_world': diameter <= 6 and avg_path_length <= 6
                }
            
        except Exception as e:
            print(f" : {e}")
            return None
    
    def detect_communities(self, method: str = 'louvain', plot: bool = True, random_seed: int = 42, 
                          standardize_ids: bool = True):
        """
        5: ID
        
        Args:
            method:  ('louvain', 'greedy_modularity')
            plot: 
            random_seed: 
            standardize_ids: ID
        """
        print("\n" + "="*50)
        print(" Step 5: Community Detection (Deterministic)")
        print("="*50)
        
        if self.largest_subgraph is None:
            print(" ")
            return None
        
        # 
        import random
        import numpy as np
        random.seed(random_seed)
        np.random.seed(random_seed)
        print(f" Random seed set to: {random_seed}")
        
        try:
            if method == 'louvain':
                import community as community_louvain
                # Louvain
                partition = community_louvain.best_partition(self.largest_subgraph, 
                                                           weight='weight', 
                                                           random_state=random_seed)
                modularity = community_louvain.modularity(partition, self.largest_subgraph, weight='weight')
            elif method == 'greedy_modularity':
                # 
                communities_generator = nx.community.greedy_modularity_communities(
                    self.largest_subgraph, weight='weight'
                )
                communities = list(communities_generator)
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
                modularity = nx.community.modularity(self.largest_subgraph, communities, weight='weight')
            else:
                raise ValueError(f": {method}")
            
            # 
            community_sizes = {}
            for node, community_id in partition.items():
                community_sizes[community_id] = community_sizes.get(community_id, 0) + 1
            
            # ID
            if standardize_ids:
                print(f" Standardizing community IDs...")
                partition, community_sizes = self._standardize_community_ids(
                    partition, community_sizes, self.largest_subgraph
                )
                print(f" Community IDs standardized by size and key members")
            
            num_communities = len(community_sizes)
            largest_community_size = max(community_sizes.values())
            
            print(f"Detection method: {method}")
            print(f"Number of communities: {num_communities}")
            print(f"Modularity: {modularity:.4f}")
            print(f"Largest community size: {largest_community_size}")
            print(f"Random seed: {random_seed} (reproducible results)")
            
            # Show top 10 largest communities
            sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop 10 community sizes (standardized IDs):")
            for i, (community_id, size) in enumerate(sorted_communities[:10]):
                print(f"  Community {community_id}: {size} nodes")
                
            # for
                if plot and len(self.largest_cc) <= 500:  # 
                    plt.figure(figsize=(12, 8))
                    
                    # spring layout
                    pos = nx.spring_layout(self.largest_subgraph, k=1, iterations=50)
                    
                    # 
                    import matplotlib.cm as cm
                    colors = cm.Set3(np.linspace(0, 1, num_communities))
                    
                    # 
                    for community_id in range(num_communities):
                        nodes_in_community = [node for node, comm in partition.items() if comm == community_id]
                        nx.draw_networkx_nodes(self.largest_subgraph, pos,
                                            nodelist=nodes_in_community,
                                            node_color=[colors[community_id]],
                                            node_size=50,
                                            alpha=0.8)
                    
                    # 
                    nx.draw_networkx_edges(self.largest_subgraph, pos, alpha=0.2, width=0.5)
                    
                    plt.title(f'Community Structure Visualization\n{num_communities} Communities, Modularity: {modularity:.4f}', 
                            fontweight='bold', fontsize=16)
                    plt.axis('off')
                    plt.show()
                
            return {
                'partition': partition,
                'modularity': modularity,
                'num_communities': num_communities,
                'community_sizes': community_sizes,
                'method': method,
                'random_seed': random_seed,
                'standardized': standardize_ids
            }
            
        except ImportError:
            print(" python-louvain: pip install python-louvain")
            return None
            
        except Exception as e:
            print(f" : {e}")
            return None
    
    def _standardize_community_ids(self, partition, community_sizes, subgraph):
        """
        ID
        
        Args:
            partition: 
            community_sizes: 
            subgraph: 
            
        Returns:
            tuple: (partition, community_sizes)
        """
        # 1. 
        community_features = {}
        
        for community_id, size in community_sizes.items():
            # 
            members = [node for node, comm_id in partition.items() if comm_id == community_id]
            
            # 
            total_weighted_degree = 0
            member_names = []
            
            for member in members:
                # 
                weighted_degree = sum(data.get('weight', 1) for _, _, data in subgraph.edges(member, data=True))
                total_weighted_degree += weighted_degree
                member_names.append(member)
            
            # ""
            member_degrees = []
            for member in members:
                degree = sum(data.get('weight', 1) for _, _, data in subgraph.edges(member, data=True))
                member_degrees.append((member, degree))
            
            member_degrees.sort(key=lambda x: x[1], reverse=True)
            
            # 
            representative = member_degrees[0][0] if member_degrees else ""
            
            community_features[community_id] = {
                'size': size,
                'total_weighted_degree': total_weighted_degree,
                'avg_weighted_degree': total_weighted_degree / size if size > 0 else 0,
                'representative': representative,
                'members': sorted(member_names)  # 
            }
        
        # 2. 
        def community_sort_key(item):
            comm_id, features = item
            return (
                -features['size'],  # 
                -features['total_weighted_degree'],  # 
                features['representative']  # 
            )
        
        sorted_communities = sorted(community_features.items(), key=community_sort_key)
        
        # 3. ID0, 1, 2, ...
        old_to_new_id = {}
        for new_id, (old_id, features) in enumerate(sorted_communities):
            old_to_new_id[old_id] = new_id
        
        # 4. partition
        new_partition = {}
        for node, old_comm_id in partition.items():
            new_partition[node] = old_to_new_id[old_comm_id]
        
        # 5. community_sizes
        new_community_sizes = {}
        for old_id, size in community_sizes.items():
            new_id = old_to_new_id[old_id]
            new_community_sizes[new_id] = size
        
        # 
        print(f"    Community ID mapping:")
        for new_id, (old_id, features) in enumerate(sorted_communities[:10]):  # 10
            representative = features['representative']
            display_rep = representative if len(representative) <= 30 else representative[:27] + "..."
            print(f"     {old_id:2d}  {new_id:2d} | Size: {features['size']:3d} | Rep: {display_rep}")
        
        if len(sorted_communities) > 10:
            print(f"     ... and {len(sorted_communities) - 10} more communities")
        
        return new_partition, new_community_sizes
    
    def save_community_results(self, communities_result, filename: str = "community_results.json"):
        """
        
        
        Args:
            communities_result: 
            filename: 
        """
        import json
        import os
        
        if communities_result is None:
            print(" No community results to save")
            return
        
        # 
        save_data = {
            'partition': communities_result['partition'],
            'modularity': communities_result['modularity'],
            'num_communities': communities_result['num_communities'],
            'community_sizes': communities_result['community_sizes'],
            'method': communities_result['method'],
            'random_seed': communities_result.get('random_seed', None),
            'standardized': communities_result.get('standardized', False),
            'timestamp': str(pd.Timestamp.now())
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f" Community results saved to: {filename}")
        except Exception as e:
            print(f" Failed to save community results: {e}")
    
    def load_community_results(self, filename: str = "community_results.json"):
        """
        
        
        Args:
            filename: 
            
        Returns:
            dict: None
        """
        import json
        import os
        
        if not os.path.exists(filename):
            print(f" File not found: {filename}")
            return None
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                communities_result = json.load(f)
            
            print(f" Community results loaded from: {filename}")
            print(f"   - Communities: {communities_result['num_communities']}")
            print(f"   - Modularity: {communities_result['modularity']:.4f}")
            print(f"   - Method: {communities_result['method']}")
            print(f"   - Random seed: {communities_result.get('random_seed', 'Not set')}")
            print(f"   - Standardized: {communities_result.get('standardized', False)}")
            print(f"   - Timestamp: {communities_result.get('timestamp', 'Unknown')}")
            
            return communities_result
            
        except Exception as e:
            print(f" Failed to load community results: {e}")
            return None
    
    def visualize_communities_with_cosmograph(self, communities_result, save_path: str = None):
        """
        Use cosmograph to create stunning community visualization
        
        Args:
            communities_result: Result from detect_communities method
            save_path: Optional path to save the visualization
        """
        try:
            import pandas as pd
            from cosmograph import cosmo
        except ImportError:
            print(" Need to install cosmograph: pip install cosmograph")
            return None
        
        if communities_result is None:
            print(" No community detection results provided")
            return None
            
        print(" Creating cosmograph community visualization...")
        
        partition = communities_result['partition']
        
        # Prepare nodes data with community colors
        nodes_data = []
        for node in self.largest_subgraph.nodes():
            if node in partition:  # Skip isolated nodes
                community_id = partition[node]
                
                # Calculate node metrics
                weighted_degree = sum(data.get('weight', 1) for _, _, data in self.largest_subgraph.edges(node, data=True))
                
                nodes_data.append({
                    'id': str(node),
                    'label': str(node),
                    'community': community_id,
                    'size': max(5, min(50, weighted_degree * 10)),  # Scale size based on weighted degree
                    'weighted_degree': weighted_degree
                })
        
        # Prepare edges data
        edges_data = []
        for edge in self.largest_subgraph.edges(data=True):
            source, target, data = edge
            if source in partition and target in partition:  # Only include edges between community members
                edges_data.append({
                    'source': str(source),
                    'target': str(target),
                    'weight': data.get('weight', 1)
                })
        
        # Create DataFrames
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        print(f" Visualization data prepared:")
        print(f"   - Nodes: {len(nodes_df):,} (excluding isolated nodes)")
        print(f"   - Edges: {len(edges_df):,}")
        print(f"   - Communities: {len(nodes_df['community'].unique())}")
        
        # Create cosmograph visualization using the correct API
        cosmos = cosmo(
            points=nodes_df,
            links=edges_df,
            point_id_by='id',
            link_source_by='source',
            link_target_by='target',
            point_color_by='community',
            point_size_by='size',
            point_label_by='label',
            link_width_by='weight',
            point_include_columns=['community', 'size', 'weighted_degree'],
            link_include_columns=['weight']
        )
        
        if save_path:
            print(f" Saving cosmograph to: {save_path}")
            # Note: cosmograph saves are handled differently, this is a placeholder
        
        print(" Cosmograph community visualization created!")
        print(" Features:")
        print("   - Dark sci-fi theme with purple accents")
        print("   - Node size  weighted degree")
        print("   - Node color = community membership")
        print("   - Interactive exploration enabled")
        
        return cosmos
    
    def visualize_top_community_members_cosmograph(self, communities_result, top_n: int = 3, save_path: str = None):
        """
        Create cosmograph visualization showing only top N members from each community
        
        Args:
            communities_result: Result from detect_communities method
            top_n: Number of top members to show per community (default: 3)
            save_path: Optional path to save the visualization
            
        Returns:
            Cosmograph visualization with filtered top community members
        """
        try:
            import pandas as pd
            from cosmograph import cosmo
        except ImportError:
            print(" Need to install cosmograph: pip install cosmograph")
            return None
        
        if communities_result is None:
            print(" No community detection results provided")
            return None
            
        print(f" Creating filtered cosmograph: Top {top_n} members per community...")
        
        partition = communities_result['partition']
        community_sizes = communities_result['community_sizes']
        
        # Calculate weighted degrees for all nodes
        node_weighted_degrees = {}
        for node in self.largest_subgraph.nodes():
            if node in partition:
                weighted_degree = sum(data.get('weight', 1) for _, _, data in self.largest_subgraph.edges(node, data=True))
                node_weighted_degrees[node] = weighted_degree
        
        # Group nodes by community and find top N members in each
        community_top_members = {}
        for community_id in community_sizes.keys():
            # Get all nodes in this community with their weighted degrees
            community_nodes = [(node, degree) for node, degree in node_weighted_degrees.items() 
                             if partition[node] == community_id]
            
            # Sort by weighted degree (descending) and take top N
            community_nodes.sort(key=lambda x: x[1], reverse=True)
            top_members = community_nodes[:top_n]
            community_top_members[community_id] = top_members
            
            print(f"   Community {community_id}: {len(community_nodes)} total  {len(top_members)} top members")
        
        # Prepare filtered nodes data
        filtered_nodes = set()
        nodes_data = []
        node_id_counter = 0
        
        for community_id, top_members in community_top_members.items():
            for node, weighted_degree in top_members:
                filtered_nodes.add(node)
                
                nodes_data.append({
                    'id': str(node_id_counter),
                    'original_name': node,
                    'label': str(node),
                    'community': community_id,
                    'size': max(10, min(80, weighted_degree * 15)),  # Scale size for better visibility
                    'weighted_degree': weighted_degree,
                    'rank_in_community': top_members.index((node, weighted_degree)) + 1
                })
                node_id_counter += 1
        
        # Create node name to ID mapping
        name_to_id = {node['original_name']: node['id'] for node in nodes_data}
        
        # Prepare filtered edges data (only edges between selected nodes)
        edges_data = []
        for edge in self.largest_subgraph.edges(data=True):
            source, target, data = edge
            if source in filtered_nodes and target in filtered_nodes:
                edges_data.append({
                    'source': name_to_id[source],
                    'target': name_to_id[target],
                    'weight': data.get('weight', 1)
                })
        
        # Create DataFrames
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        print(f" Filtered visualization data prepared:")
        print(f"   - Original nodes: {len(node_weighted_degrees):,}")
        print(f"   - Filtered nodes: {len(nodes_df):,} (top {top_n} per community)")
        print(f"   - Filtered edges: {len(edges_df):,}")
        print(f"   - Communities: {len(nodes_df['community'].unique())}")
        
        # Create cosmograph visualization for top community members
        cosmos = cosmo(
            points=nodes_df,
            links=edges_df,
            point_id_by='id',
            link_source_by='source',
            link_target_by='target',
            point_color_by='community',
            point_size_by='size',
            point_label_by='label',
            link_width_by='weight',
            point_include_columns=['community', 'size', 'weighted_degree', 'rank_in_community', 'original_name'],
            link_include_columns=['weight']
        )
        
        if save_path:
            print(f" Saving filtered cosmograph to: {save_path}")
        
        print(" Filtered community cosmograph created!")
        print(" Features:")
        print(f"   - Shows only top {top_n} members from each community")
        print("   - Node size  weighted degree")
        print("   - Node color = community membership")
        print("   - Cleaner visualization with key players highlighted")
        print("   - Includes rank_in_community information")
        
        # Print community summary
        print(f"\n Community Summary:")
        for community_id, top_members in community_top_members.items():
            print(f"   Community {community_id} (size: {community_sizes[community_id]}):")
            for i, (node, degree) in enumerate(top_members, 1):
                print(f"      {i}. {node} (degree: {degree:.3f})")
        
        return cosmos
        
    def create_meta_community_graph(self, communities_result):
        """
        Create a meta-graph where communities become nodes
        
        Args:
            communities_result: Result from detect_communities method
            
        Returns:
            Cosmograph visualization of community-level network
        """
        try:
            import pandas as pd
            from cosmograph import cosmo
            import networkx as nx
        except ImportError:
            print(" Need to install cosmograph and networkx")
            return None
            
        if communities_result is None:
            print(" No community detection results provided")
            return None
            
        print(" Creating meta-community network...")
        
        partition = communities_result['partition']
        community_sizes = communities_result['community_sizes']
        
        # Create meta-graph
        meta_graph = nx.Graph()
        
        # Add community nodes
        for community_id, size in community_sizes.items():
            meta_graph.add_node(community_id, size=size)
        
        # Add edges between communities (inter-community connections)
        community_connections = {}
        
        for edge in self.largest_subgraph.edges(data=True):
            source, target, data = edge
            if source in partition and target in partition:
                comm_source = partition[source]
                comm_target = partition[target]
                
                if comm_source != comm_target:  # Inter-community edge
                    edge_key = tuple(sorted([comm_source, comm_target]))
                    if edge_key not in community_connections:
                        community_connections[edge_key] = 0
                    community_connections[edge_key] += data.get('weight', 1)
        
        # Add weighted edges to meta-graph
        for (comm1, comm2), weight in community_connections.items():
            meta_graph.add_edge(comm1, comm2, weight=weight)
        
        # Prepare data for cosmograph
        meta_nodes_data = []
        for community_id in meta_graph.nodes():
            size = community_sizes[community_id]
            
            meta_nodes_data.append({
                'id': f'Community_{community_id}',
                'label': f'C{community_id}\n({size} nodes)',
                'size': max(20, min(200, size * 2)),  # Scale based on community size
                'community_size': size,
                'community_id': community_id
            })
        
        meta_edges_data = []
        for edge in meta_graph.edges(data=True):
            comm1, comm2, data = edge
            meta_edges_data.append({
                'source': f'Community_{comm1}',
                'target': f'Community_{comm2}',
                'weight': data['weight']
            })
        
        # Create DataFrames
        meta_nodes_df = pd.DataFrame(meta_nodes_data)
        meta_edges_df = pd.DataFrame(meta_edges_data)
        
        print(f"  Meta-network constructed:")
        print(f"   - Community nodes: {len(meta_nodes_df)}")
        print(f"   - Inter-community connections: {len(meta_edges_df)}")
        
        # Create meta-community cosmograph using correct API
        meta_cosmos = cosmo(
            points=meta_nodes_df,
            links=meta_edges_df,
            point_id_by='id',
            link_source_by='source',
            link_target_by='target',
            point_size_by='size',
            point_label_by='label',
            link_width_by='weight',
            point_include_columns=['community_size', 'community_id'],
            link_include_columns=['weight']
        )
        
        print(" Meta-community cosmograph created!")
        print(" Styling:")
        print("   - Dark navy background")
        print("   - Red community nodes")
        print("   - Golden inter-community links")
        print("   - Node size  community size")
        
        return meta_cosmos
    
    def analyze_community_temporal_evolution(self, target_communities: List[int], top_n_leaders: int = 5,
                                           decades: List[str] = None, save_figures: bool = True, 
                                           figures_dir: str = "Figures"):
        """
         - 
        
        Args:
            target_communities: ID (e.g., [17, 20, 0, 18, 26])
            top_n_leaders: 
            decades: 
            save_figures: 
            figures_dir: 
            
        Returns:
            Dict: 
        """
        print("\n" + "="*60)
        print(" Community Temporal Evolution Analysis (Decade-Grouped)")
        print("="*60)
        
        if decades is None:
            decades = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
        
        # Create figures directory
        if save_figures:
            import os
            os.makedirs(figures_dir, exist_ok=True)
        
        # 
        current_communities = self.detect_communities(method='louvain', plot=False, 
                                                     random_seed=42, standardize_ids=True)
        if not current_communities:
            print(" Failed to detect communities")
            return None
        
        # 
        community_labels = self._create_community_labels(target_communities, current_communities)
        
        # Initialize results
        results = {
            'target_communities': target_communities,
            'community_labels': community_labels,
            'current_communities': current_communities,  #  
            'decade_analysis': {},  # 
            'structure_summary': [],  # 
            'community_leaders': {},  # 
            'figures_saved': []
        }
        
        print(f" Analyzing {len(target_communities)} target communities across {len(decades)} decades")
        print(f" Showing top {top_n_leaders} leaders per community per decade")
        
        # 
        for decade in decades:
            print(f"\n" + "="*50)
            print(f" Decade: {decade}")
            print("="*50)
            
            decade_result = self._analyze_decade_communities(
                decade, target_communities, community_labels, 
                top_n_leaders, current_communities
            )
            
            if decade_result:
                results['decade_analysis'][decade] = decade_result
                
                # 
                self._print_decade_community_structure(decade, decade_result)
                
                # 
                self._print_decade_community_leaders(decade, decade_result, top_n_leaders)
        
        # 
        self._create_structure_summary(results)
        
        # 
        self._collect_community_leaders(results)
        
        # 
        if save_figures:
            self._visualize_decade_evolution(results, figures_dir)
            self._visualize_community_evolution_tree(results, figures_dir)
        
        return results
    
    def _collect_community_leaders(self, results: Dict):
        """
        
        """
        print(f"\n Collecting Community Leaders Information...")
        
        community_leaders = {}
        
        for comm_id in results['target_communities']:
            community_leaders[comm_id] = {
                'label': results['community_labels'][comm_id],
                'decade_leaders': {}
            }
            
            total_unique_leaders = set()
            
            for decade, decade_data in results['decade_analysis'].items():
                comm_data = decade_data['communities'].get(comm_id, {})
                leaders = comm_data.get('leaders', [])
                
                community_leaders[comm_id]['decade_leaders'][decade] = leaders
                
                # 
                for leader_info in leaders:
                    name = leader_info[0] if leader_info else None
                    if name:
                        total_unique_leaders.add(name)
            
            community_leaders[comm_id]['total_unique_leaders'] = len(total_unique_leaders)
            community_leaders[comm_id]['all_leaders'] = list(total_unique_leaders)
            
            print(f"   Community {comm_id}: {len(total_unique_leaders)} unique leaders across all decades")
        
        results['community_leaders'] = community_leaders
    
    def _create_community_labels(self, target_communities: List[int], current_communities: Dict) -> Dict:
        """
        
        """
        print(f"\n Creating Community Academic Labels")
        print("="*50)
        
        partition = current_communities['partition']
        community_labels = {}
        
        for comm_id in target_communities:
            # 
            community_members = [node for node, c_id in partition.items() if c_id == comm_id]
            
            if not community_members:
                community_labels[comm_id] = f"Community {comm_id} (Unknown)"
                continue
            
            # 
            research_keywords = {}
            for member in community_members[:20]:  # 20
                papers = self.network.nodes[member].get('papers', [])
                for paper in papers:
                    title = paper.get('title', '').lower()
                    # 
                    if 'urban' in title:
                        research_keywords['Urban Planning'] = research_keywords.get('Urban Planning', 0) + 1
                    if 'transport' in title or 'traffic' in title:
                        research_keywords['Transportation'] = research_keywords.get('Transportation', 0) + 1
                    if 'model' in title or 'simulation' in title:
                        research_keywords['Modeling'] = research_keywords.get('Modeling', 0) + 1
                    if 'network' in title:
                        research_keywords['Network Analysis'] = research_keywords.get('Network Analysis', 0) + 1
                    if 'data' in title or 'gis' in title:
                        research_keywords['Data Science'] = research_keywords.get('Data Science', 0) + 1
                    if 'spatial' in title or 'geographic' in title:
                        research_keywords['Spatial Analysis'] = research_keywords.get('Spatial Analysis', 0) + 1
                    if 'environment' in title or 'sustainability' in title:
                        research_keywords['Environment'] = research_keywords.get('Environment', 0) + 1
                    if 'policy' in title or 'governance' in title:
                        research_keywords['Policy'] = research_keywords.get('Policy', 0) + 1
            
            # 
            if research_keywords:
                main_field = max(research_keywords.items(), key=lambda x: x[1])[0]
                community_labels[comm_id] = f"Community {comm_id} ({main_field})"
            else:
                community_labels[comm_id] = f"Community {comm_id} (General)"
            
            print(f"   Community {comm_id}: {community_labels[comm_id]} | {len(community_members)} members")
        
        return community_labels
    
    def _analyze_decade_communities(self, decade: str, target_communities: List[int], 
                                  community_labels: Dict, top_n_leaders: int, 
                                  current_communities: Dict) -> Dict:
        """
        
        """
        # 
        decade_authors = set()
        decade_papers_info = []
        
        for node, data in self.network.nodes(data=True):
            papers = data.get('papers', [])
            node_decade_papers = []
            for paper in papers:
                if paper.get('decade') == decade:
                    decade_authors.add(node)
                    node_decade_papers.append(paper)
                    decade_papers_info.append({
                        'author': node,
                        'title': paper.get('title', ''),
                        'year': paper.get('year', '')
                    })
            
            # 
            if node_decade_papers:
                self.network.nodes[node][f'{decade}_papers'] = node_decade_papers
        
        if not decade_authors:
            print(f"    No authors found for {decade}")
            return None
        
        # 
        partition = current_communities['partition']
        decade_result = {
            'total_authors': len(decade_authors),
            'total_papers': len(decade_papers_info),
            'communities': {}
        }
        
        for comm_id in target_communities:
            # 
            community_members = [node for node, c_id in partition.items() if c_id == comm_id]
            decade_active_members = [member for member in community_members if member in decade_authors]
            
            if not decade_active_members:
                decade_result['communities'][comm_id] = {
                    'label': community_labels[comm_id],
                    'active_members': 0,
                    'papers_count': 0,
                    'leaders': [],
                    'collaboration_density': 0
                }
                continue
            
            # 
            community_papers_count = 0
            member_paper_counts = {}
            
            for member in decade_active_members:
                member_papers = self.network.nodes[member].get(f'{decade}_papers', [])
                community_papers_count += len(member_papers)
                member_paper_counts[member] = len(member_papers)
            
            # 
            internal_collaborations = 0
            possible_collaborations = len(decade_active_members) * (len(decade_active_members) - 1) / 2
            
            for i, member1 in enumerate(decade_active_members):
                for member2 in decade_active_members[i+1:]:
                    if self.network.has_edge(member1, member2):
                        internal_collaborations += 1
            
            collaboration_density = internal_collaborations / possible_collaborations if possible_collaborations > 0 else 0
            
            # 
            member_scores = []
            for member in decade_active_members:
                papers_score = member_paper_counts.get(member, 0)
                degree_score = sum(data.get('weight', 1) for _, _, data in self.network.edges(member, data=True))
                combined_score = papers_score * 2 + degree_score  # 
                member_scores.append((member, combined_score, papers_score, degree_score))
            
            # 
            member_scores.sort(key=lambda x: x[1], reverse=True)
            leaders = member_scores[:top_n_leaders]
            
            decade_result['communities'][comm_id] = {
                'label': community_labels[comm_id],
                'active_members': len(decade_active_members),
                'papers_count': community_papers_count,
                'leaders': leaders,
                'collaboration_density': collaboration_density
            }
        
        return decade_result
    
    def _print_decade_community_structure(self, decade: str, decade_result: Dict):
        """
        
        """
        print(f"\n {decade} Community Structure Data:")
        print(f"   Total Authors: {decade_result['total_authors']}")
        print(f"   Total Papers: {decade_result['total_papers']}")
        print()
        
        for comm_id, comm_data in decade_result['communities'].items():
            print(f"    {comm_data['label']}:")
            print(f"      Active Members: {comm_data['active_members']:3d}")
            print(f"      Papers Published: {comm_data['papers_count']:3d}")
            print(f"      Internal Collaboration Density: {comm_data['collaboration_density']:.3f}")
    
    def _print_decade_community_leaders(self, decade: str, decade_result: Dict, top_n: int):
        """
        
        """
        print(f"\n {decade} Community Leaders:")
        
        for comm_id, comm_data in decade_result['communities'].items():
            if not comm_data['leaders']:
                print(f"\n    {comm_data['label']}: No active members")
                continue
                
            print(f"\n    {comm_data['label']} - Top {min(len(comm_data['leaders']), top_n)} Leaders:")
            
            for i, (name, score, papers, degree) in enumerate(comm_data['leaders'], 1):
                # 
                display_name = name
                print(f"      {i}. {display_name:<35} | Papers: {papers:2d} | Network Score: {degree:5.2f} | Combined: {score:5.2f}")
    
    def _create_structure_summary(self, results: Dict):
        """
        
        """
        print(f"\n Structure Evolution Summary:")
        print("="*50)
        
        decades = sorted(results['decade_analysis'].keys())
        
        for comm_id in results['target_communities']:
            label = results['community_labels'][comm_id]
            print(f"\n {label} Evolution:")
            
            for decade in decades:
                if decade in results['decade_analysis']:
                    comm_data = results['decade_analysis'][decade]['communities'].get(comm_id, {})
                    members = comm_data.get('active_members', 0)
                    papers = comm_data.get('papers_count', 0)
                    density = comm_data.get('collaboration_density', 0)
                    print(f"   {decade}: {members:2d} members, {papers:3d} papers, density: {density:.3f}")
    
    def _visualize_decade_evolution(self, results: Dict, figures_dir: str):
        """
         - 2x2
        """
        print(f"\n Creating decade evolution visualizations...")
        
        decades = sorted(results['decade_analysis'].keys())
        communities = results['target_communities']
        
        if len(decades) < 2:
            print("    Need at least 2 decades for visualization")
            return
        
        #  1x2CD
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 
        community_leaders = {}
        for comm_id in communities:
            # 
            current_partition = results.get('current_communities', {}).get('partition', {})
            community_members = [node for node, c_id in current_partition.items() if c_id == comm_id]
            
            if community_members:
                # 
                member_degrees = []
                for member in community_members:
                    weighted_degree = sum(data.get('weight', 1) for _, _, data in self.network.edges(member, data=True))
                    member_degrees.append((member, weighted_degree))
                
                if member_degrees:
                    top_member = max(member_degrees, key=lambda x: x[1])
                    # 
                    name_parts = top_member[0].split(',')
                    surname = name_parts[0].strip() if name_parts else top_member[0]
                    community_leaders[comm_id] = surname[:15]  # 
                else:
                    community_leaders[comm_id] = f"C{comm_id}"
            else:
                community_leaders[comm_id] = f"C{comm_id}"
        
        # 
        from matplotlib.colors import LinearSegmentedColormap
        # A
        colors_orange_red = ['#FFF3E0', '#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726', '#FF9800', '#FB8C00', '#F57C00', '#EF6C00', '#E65100', '#D84315', '#BF360C']
        cmap_members = LinearSegmentedColormap.from_list('orange_red', colors_orange_red, N=256)
        
        # B  
        colors_blue_purple = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1', '#673AB7', '#5E35B1']
        cmap_papers = LinearSegmentedColormap.from_list('blue_purple', colors_blue_purple, N=256)
        
        # 1.  - Y
        all_member_values = []
        for comm_id in communities:
            for decade in decades:
                if decade in results['decade_analysis']:
                    count = results['decade_analysis'][decade]['communities'].get(comm_id, {}).get('active_members', 0)
                    if count > 0:
                        all_member_values.append(count)
        
        if all_member_values:
            vmin_members, vmax_members = min(all_member_values), max(all_member_values)
            
            for comm_idx, comm_id in enumerate(communities):
                x_data = []  # decade indices
                y_data = []  # fixed community positions
                size_data = []  # bubble sizes
                color_data = []  # color values
                
                for decade_idx, decade in enumerate(decades):
                    if decade in results['decade_analysis']:
                        count = results['decade_analysis'][decade]['communities'].get(comm_id, {}).get('active_members', 0)
                        if count > 0:
                            x_data.append(decade_idx)
                            y_data.append(comm_idx)  # Y
                            size_data.append(max(100, min(1500, count * 25)))
                            color_data.append(count)
                
                if x_data:
                    scatter1 = ax1.scatter(x_data, y_data, s=size_data, c=color_data, 
                                         cmap=cmap_members, vmin=vmin_members, vmax=vmax_members,
                                         alpha=0.8, edgecolors='white', linewidth=1)
        
        ax1.set_xticks(range(len(decades)))
        ax1.set_xticklabels(decades, rotation=45, fontsize=FONT_CONFIG['tick'])
        ax1.set_yticks(range(len(communities)))
        ax1.set_yticklabels([f'{community_leaders[comm_id]}' for comm_id in communities], fontsize=10)
        ax1.set_xlabel('Decade', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2.  - Y
        all_paper_values = []
        for comm_id in communities:
            for decade in decades:
                if decade in results['decade_analysis']:
                    count = results['decade_analysis'][decade]['communities'].get(comm_id, {}).get('papers_count', 0)
                    if count > 0:
                        all_paper_values.append(count)
        
        if all_paper_values:
            vmin_papers, vmax_papers = min(all_paper_values), max(all_paper_values)
            
            for comm_idx, comm_id in enumerate(communities):
                x_data = []  # decade indices
                y_data = []  # fixed community positions
                size_data = []  # bubble sizes
                color_data = []  # color values
                
                for decade_idx, decade in enumerate(decades):
                    if decade in results['decade_analysis']:
                        count = results['decade_analysis'][decade]['communities'].get(comm_id, {}).get('papers_count', 0)
                        if count > 0:
                            x_data.append(decade_idx)
                            y_data.append(comm_idx)  # Y
                            size_data.append(max(100, min(1800, count * 20)))
                            color_data.append(count)
                
                if x_data:
                    scatter2 = ax2.scatter(x_data, y_data, s=size_data, c=color_data, 
                                         cmap=cmap_papers, vmin=vmin_papers, vmax=vmax_papers,
                                         alpha=0.8, edgecolors='white', linewidth=1)
        
        ax2.set_xticks(range(len(decades)))
        ax2.set_xticklabels(decades, rotation=45, fontsize=FONT_CONFIG['tick'])
        ax2.set_yticks(range(len(communities)))
        ax2.set_yticklabels([f'{community_leaders[comm_id]}' for comm_id in communities], fontsize=10)
        ax2.set_xlabel('Decade', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3.  ( - C)
        # total_authors = []
        # for decade in decades:
        #     if decade in results['decade_analysis']:
        #         total_authors.append(results['decade_analysis'][decade]['total_authors'])
        #     else:
        #         total_authors.append(0)
        
        # ax3.plot(decades, total_authors, 'o-', linewidth=3, markersize=8, 
        #         color=JOURNAL_COLORS['primary'])
        # ax3.set_ylabel('Total Authors', fontweight='bold')
        # ax3.grid(True, alpha=0.3)
        # ax3.tick_params(axis='x', rotation=45)
        
        # # 
        # for i, count in enumerate(total_authors):
        #     ax3.annotate(f'{count}', (i, count), textcoords="offset points", 
        #                 xytext=(0,10), ha='center', fontweight='bold', fontsize=9)
        
        # 4.  ( - D)
        # total_papers = []
        # for decade in decades:
        #     if decade in results['decade_analysis']:
        #         total_papers.append(results['decade_analysis'][decade]['total_papers'])
        #     else:
        #         total_papers.append(0)
        
        # ax4.plot(decades, total_papers, 's-', linewidth=3, markersize=8, 
        #         color=JOURNAL_COLORS['secondary'])
        # ax4.set_ylabel('Total Papers', fontweight='bold')
        # ax4.grid(True, alpha=0.3)
        # ax4.tick_params(axis='x', rotation=45)
        
        # # 
        # for i, count in enumerate(total_papers):
        #     ax4.annotate(f'{count}', (i, count), textcoords="offset points", 
        #                 xytext=(0,10), ha='center', fontweight='bold', fontsize=9)
        
        #  colorbar
        if all_member_values:
            # Acolorbar - 
            sm1 = plt.cm.ScalarMappable(cmap=cmap_members, norm=plt.Normalize(vmin=vmin_members, vmax=vmax_members))
            sm1.set_array([])
            cbar1 = plt.colorbar(sm1, ax=ax1, orientation='vertical', shrink=0.8, pad=0.02)
            cbar1.set_label('Active Members', fontweight='bold', fontsize=11)
            
        if all_paper_values:
            # Bcolorbar - 
            sm2 = plt.cm.ScalarMappable(cmap=cmap_papers, norm=plt.Normalize(vmin=vmin_papers, vmax=vmax_papers))
            sm2.set_array([])
            cbar2 = plt.colorbar(sm2, ax=ax2, orientation='vertical', shrink=0.8, pad=0.02)
            cbar2.set_label('Papers Published', fontweight='bold', fontsize=11)
        
        # 
        ax1.text(0.5, 1.02, 'a. Active Members Over Time', transform=ax1.transAxes, 
                ha='center', fontweight='bold', fontsize=FONT_CONFIG['title'])
        ax2.text(0.5, 1.02, 'b. Papers Published Over Time', transform=ax2.transAxes, 
                ha='center', fontweight='bold', fontsize=FONT_CONFIG['title'])
        # ax3.text(0.5, 1.02, 'C. Total Authors Growth', transform=ax3.transAxes, 
        #         ha='center', fontweight='bold', fontsize=14)
        # ax4.text(0.5, 1.02, 'D. Total Papers Growth', transform=ax4.transAxes, 
        #         ha='center', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        # 
        figure_path = os.path.join(figures_dir, '02-community_decade_evolution.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
        results['figures_saved'].append(figure_path)
        print(f" Saved: {figure_path}")
        
        plt.show()
        
    def _find_batty_community(self, results: Dict) -> int:
        """
        Michael Batty
        """
        current_partition = results.get('current_communities', {}).get('partition', {})
        
        if not current_partition:
            print(" No current_communities partition found")
            return None
        
        print(f" Searching for Michael Batty in {len(current_partition)} authors...")
        
        # Michael Batty
        batty_variations = ['Batty, Michael', 'Batty, M', 'Michael Batty', 'Batty,Michael', 'Batty,M.']
        batty_community = None
        found_name = None
        
        for node in current_partition:
            for variation in batty_variations:
                if variation.lower() in node.lower():
                    batty_community = current_partition[node]
                    found_name = node
                    print(f" Found Michael Batty in Community {batty_community}: {node}")
                    break
            if batty_community is not None:
                break
        
        # Batty
        if batty_community is None:
            print(" Michael Batty not found with standard variations")
            print(" All authors with 'Batty' in name:")
            batty_authors = [(name, current_partition[name]) for name in current_partition.keys() if 'batty' in name.lower()]
            for author, comm_id in batty_authors[:10]:  # 10
                print(f"   - {author} (Community {comm_id})")
            
            # Batty
            if batty_authors:
                batty_community = batty_authors[0][1]
                found_name = batty_authors[0][0]
                print(f" Using first Batty match: {found_name} in Community {batty_community}")
        
        if batty_community is not None:
            print(f" Michael Batty located: {found_name} in Community {batty_community}")
        else:
            print(" No Batty found, returning None")
            
        return batty_community
    
    def _select_complete_lineage_communities(self, results: Dict) -> List[int]:
        """
        
        """
        decades = sorted(results['decade_analysis'].keys())
        community_completeness = {}
        
        # 
        for comm_id in results['target_communities']:
            active_decades = 0
            total_members = 0
            
            for decade in decades:
                if decade in results['decade_analysis']:
                    comm_data = results['decade_analysis'][decade]['communities'].get(comm_id, {})
                    leaders = comm_data.get('leaders', [])
                    if leaders:
                        active_decades += 1
                        total_members += len(leaders)
            
            #  =  + 
            completeness_score = active_decades * 2 + total_members * 0.1
            community_completeness[comm_id] = {
                'score': completeness_score,
                'active_decades': active_decades,
                'total_members': total_members
            }
        
        # Michael Battytarget_communities
        batty_community = self._find_batty_community(results)
        
        # Batty
        sorted_communities = sorted(community_completeness.items(), 
                                  key=lambda x: x[1]['score'], reverse=True)
        
        selected_communities = []
        
        #  Battytarget_communities
        if batty_community is not None and batty_community in community_completeness:
            selected_communities.append(batty_community)
            print(f"    Found Michael Batty in Community {batty_community} (included)")
        elif batty_community is not None:
            print(f"    Michael Batty found in Community {batty_community}, but not in target communities (skipped)")
        
        # 3
        for comm_id, stats in sorted_communities:
            if comm_id not in selected_communities and len(selected_communities) < 3:
                selected_communities.append(comm_id)
        
        print(f" Selected communities with complete lineages:")
        for comm_id in selected_communities:
            #  IDcommunity_completeness
            if comm_id in community_completeness:
                stats = community_completeness[comm_id]
                print(f"   Community {comm_id}: {stats['active_decades']} active decades, {stats['total_members']} total members")
            else:
                print(f"   Community {comm_id}: (no temporal data available)")
        
        return selected_communities

    def _visualize_community_evolution_tree(self, results: Dict, figures_dir: str):
        """
         - Michael Batty
        
        
        Args:
            results: 
            figures_dir: 
        """
        print("\n Creating interactive community evolution network...")
        
        #  Michael Batty
        batty_community = self._find_batty_community(results)
        if batty_community is None:
            print(" Michael Batty not found in any community")
            return None
        
        print(f" Found Michael Batty in Community {batty_community}")
        
        decades = sorted(results['decade_analysis'].keys())
        
        #  
        network_data = self._build_community_network_data(results, batty_community, decades)
        
        if not network_data['nodes']:
            print(" No valid authors found for the community")
            return None
        
        #  
        if not network_data['links']:
            print(" No internal collaborations found, but showing community member distribution")
            print("    This could indicate the community is formed by:")
            print("      - Indirect connections through other authors")
            print("      - Similar research topics/keywords")
            print("      - Methodological or institutional similarity")
            # 
        
        print(f" Community {batty_community}: {len(network_data['nodes'])} authors, {len(network_data['links'])} collaborations")
        
        #  
        html_content = self._create_community_network_html(network_data, batty_community, decades)
        
        # HTML
        figure_path = os.path.join(figures_dir, "06-community-evolution-network.html")
        with open(figure_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        results['figures_saved'].append(figure_path)
        print(f" Saved: {figure_path}")
        print(" Open the HTML file in your browser to view the interactive network")
        
        return network_data
    
    def _build_community_network_data(self, results: Dict, community_id: int, decades: List[str]) -> Dict:
        """
         - 
        
        Args:
            results: 
            community_id: ID
            decades: 
            
        Returns:
            Dict: nodeslinks
        """
        print(f" Building network data for Community {community_id}...")
        
        #   - ECharts
        decade_colors = {
            '1970s': '#5470c6',  # 
            '1980s': '#91cc75',  # 
            '1990s': '#fac858',  # 
            '2000s': '#ee6666',  # 
            '2010s': '#73c0de',  # 
            '2020s': '#3ba272',  # 
            'unknown': '#95a5a6'  #  - 
        }
        
        #  Step 1: 
        author_first_decade = {}  #  -> 
        author_info = {}  # 
        
        # 
        current_partition = results.get('current_communities', {}).get('partition', {})
        if not current_partition:
            print(" No current_communities partition data found in results")
            print(f" Available keys in results: {list(results.keys())}")
            return {'nodes': [], 'links': []}
        
        all_community_members = [node for node, c_id in current_partition.items() if c_id == community_id]
        print(f"    Found {len(all_community_members)} total members in Community {community_id}")
        
        #  
        for member in all_community_members:
            if member not in author_first_decade:
                # 
                author_first_decade[member] = 'unknown'
                node_data = self.network.nodes.get(member, {})
                author_info[member] = {
                    'name': member,
                    'first_decade': 'unknown',  #  first_decade
                    'decade_color': '#95a5a6',  #  decade_color
                    'papers': node_data.get('paper_count', 0),
                    'degree': self.network.degree(member),
                    'weighted_degree': sum(self.network[member][neighbor].get('weight', 1) 
                                        for neighbor in self.network.neighbors(member))
                }
        
        # 
        for decade in sorted(decades):
            if decade not in results['decade_analysis']:
                continue
            
            decade_data = results['decade_analysis'][decade]
            if 'communities' not in decade_data or community_id not in decade_data['communities']:
                continue
            
            decade_community_data = decade_data['communities'][community_id]
            
            #  leaders
            decade_active_authors = set()
            
            # leaders
            if 'leaders' in decade_community_data:
                for leader_info in decade_community_data['leaders']:
                    if isinstance(leader_info, (list, tuple)) and len(leader_info) >= 2:
                        author_name = leader_info[0]  # (author_name, score, paper_count)
                        decade_active_authors.add(author_name)
            
            #  
            for member in all_community_members:
                # 
                node_data = self.network.nodes.get(member, {})
                decade_papers = node_data.get(f'{decade}_papers', [])
                if decade_papers:  # 
                    decade_active_authors.add(member)
            
            # 
            for author_name in decade_active_authors:
                if author_name not in author_first_decade:
                    author_first_decade[author_name] = decade
                    
                    # 
                    node_data = self.network.nodes.get(author_name, {})
                    author_info[author_name] = {
                        'name': author_name,
                        'first_decade': decade,
                        'decade_color': decade_colors.get(decade, '#808080'),
                        'degree': self.network.degree(author_name) if self.network.has_node(author_name) else 1,
                        'papers': len(node_data.get('papers', [])),
                    }
                else:
                    #  first_decadedecade_color
                    if author_name in author_info and 'first_decade' not in author_info[author_name]:
                        author_info[author_name]['first_decade'] = author_first_decade[author_name]
                        author_info[author_name]['decade_color'] = decade_colors.get(author_first_decade[author_name], '#808080')
        
        print(f"    Found {len(author_first_decade)} unique authors across {len(decades)} decades")
        
        #  Step 2: 
        nodes = []
        for author_name, author_data in author_info.items():
            first_decade = author_data['first_decade']
            
            # 
            if ',' in author_name:
                #  "Batty, Michael" -> "Batty"
                display_name = author_name.split(',')[0].strip()
            elif ' ' in author_name:
                #  "Michael Batty" -> "Batty"
                name_parts = author_name.split()
                display_name = name_parts[-1] if len(name_parts) > 1 else name_parts[0]
            else:
                # 
                display_name = author_name
            
            node = {
                'id': author_name,
                'name': display_name,  #  
                'full_name': author_name,  #  
                'first_decade': first_decade,
                'color': author_data['decade_color'],
                'size': min(max(author_data.get('degree', 1) * 4, 12), 35),  #  
                'papers': author_data.get('papers', 0),
                'degree': author_data.get('degree', 1),
                'symbolSize': min(max(author_data.get('degree', 1) * 4, 12), 35)  #  
            }
            nodes.append(node)
        
        #  Step 3: 
        links = []
        collaboration_count = {}  # 
        
        #  results
        G = self.network
        print(f"    Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        #  Debug: 
        missing_authors = []
        existing_authors = []
        for author in author_first_decade.keys():
            if G.has_node(author):
                existing_authors.append(author)
            else:
                missing_authors.append(author)
        
        print(f"    Author status: {len(existing_authors)} in network, {len(missing_authors)} missing")
        if missing_authors and len(missing_authors) <= 5:
            print(f"    Missing authors: {missing_authors}")
        
        #  Debug: 
        internal_edges = 0
        cross_community_edges = 0
        total_possible_edges = len(existing_authors) * (len(existing_authors) - 1) // 2
        
        # 
        for author1 in author_first_decade.keys():
            for author2 in author_first_decade.keys():
                if author1 >= author2:  # 
                    continue
                    
                # 
                if G.has_edge(author1, author2):
                    internal_edges += 1
                    edge_data = G[author1][author2]
                    weight = edge_data.get('weight', 1)
                    
                    # 
                    decade1 = author_first_decade[author1]
                    decade2 = author_first_decade[author2]
                    
                    #  
                    edge_color = '#808080'  # 
                    edge_type = 'intra-decade' if decade1 == decade2 else 'inter-decade'
                    
                    link = {
                        'source': author1,
                        'target': author2,
                        'weight': weight,
                        'color': edge_color,
                        'type': edge_type,
                        'decades': f"{decade1}  {decade2}" if decade1 != decade2 else decade1,
                        'lineStyle': {
                            'color': edge_color,
                            'width': min(max(weight * 2, 2), 6),  #  12
                            'opacity': 0.7 if edge_type == 'intra-decade' else 0.9
                        }
                    }
                    links.append(link)
        
        #  Debug: 
        print(f"    Found {internal_edges} internal edges out of {total_possible_edges} possible")
        print(f"    Community connectivity: {internal_edges/total_possible_edges:.1%}" if total_possible_edges > 0 else "    No possible connections")
        print(f"    Found {len(links)} collaboration relationships for visualization")
        
        # 
        inter_decade_count = sum(1 for link in links if link['type'] == 'inter-decade')
        intra_decade_count = len(links) - inter_decade_count
        print(f"        {intra_decade_count} intra-decade, {inter_decade_count} inter-decade collaborations")
        
        return {
            'nodes': nodes,
            'links': links,
            'community_id': community_id,
            'decades': decades,
            'decade_colors': decade_colors,
            'stats': {
                'total_authors': len(nodes),
                'total_collaborations': len(links),
                'inter_decade_collaborations': inter_decade_count,
                'intra_decade_collaborations': intra_decade_count
            }
        }
   
    def _create_community_network_html(self, network_data: Dict, community_id: int, decades: List[str]) -> str:
        """
        EChartsHTML
        
        Args:
            network_data: 
            community_id: ID
            decades: 
            
        Returns:
            str: HTML
        """
        print("    Creating interactive ECharts network visualization...")
        
        nodes = network_data['nodes']
        links = network_data['links']
        decade_colors = network_data['decade_colors']
        stats = network_data['stats']
        
        # 
        decade_stats = {}
        for node in nodes:
            decade = node['first_decade']
            if decade not in decade_stats:
                decade_stats[decade] = 0
            decade_stats[decade] += 1
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EPB Community {community_id} Evolution Network - Michael Batty's Academic Community</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #2c3e50;
        }}
        .container {{
            max-width: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .main-content {{
            display: flex;
            gap: 20px;
        }}
        .chart-section {{
            flex: 1;
        }}
        .legend-section {{
            width: 280px;
            min-width: 280px;
        }}
        .title {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .subtitle {{
            font-size: 16px;
            color: #7f8c8d;
            margin-bottom: 20px;
        }}
        .stats-panel {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .stat-card {{
            text-align: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            border: 1px solid rgba(52, 152, 219, 0.2);
        }}
        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .legend-panel {{
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .legend-category {{
            margin-bottom: 15px;
        }}
        .legend-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .legend-items {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 6px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 14px;
            color: #2c3e50;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
            border: 1px solid rgba(44, 62, 80, 0.2);
        }}
        #network-chart {{
            width: 100%;
            height: 800px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        .controls {{
            margin: 20px 0;
            text-align: center;
        }}
        .control-button {{
            padding: 10px 20px;
            margin: 0 5px;
            background: #3498db;
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .control-button:hover {{
            background: #2980b9;
            transform: translateY(-2px);
        }}
        .control-button.active {{
            background: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title"> EPB Community {community_id} Evolution Network</div>
            <div class="subtitle">Michael Batty's Academic Community - Generational Collaboration Analysis</div>
        </div>
        
        <div class="stats-panel">
            <div class="stat-card">
                <div class="stat-number">{stats['total_authors']}</div>
                <div class="stat-label">Total Authors</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['total_collaborations']}</div>
                <div class="stat-label">Collaborations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['inter_decade_collaborations']}</div>
                <div class="stat-label">Cross-Decade Links</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(decades)}</div>
                <div class="stat-label">Active Decades</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="control-button active" onclick="showAllNodes()">Show All</button>
            <button class="control-button" onclick="showInterDecadeOnly()">Cross-Decade Only</button>
            <button class="control-button" onclick="focusBatty()">Focus on Batty</button>
            <button class="control-button" onclick="resetZoom()">Reset View</button>
            <button class="control-button" onclick="saveAsImage()"> Save PNG</button>
        </div>
        
        <div class="main-content">
            <div class="chart-section">
                <div id="network-chart"></div>
            </div>
            
            <div class="legend-section">
                <div class="legend-panel">
                    <div class="legend-category">
                        <div class="legend-title"> Author Generations</div>
                        <div class="legend-items">"""
        
        # 
        for decade in sorted(decades):
            if decade in decade_colors:
                count = decade_stats.get(decade, 0)
                html_content += f'''
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: {decade_colors[decade]}"></div>
                                <span>{decade} ({count})</span>
                            </div>'''
        
        # 
        
        html_content += """
                        </div>
                    </div>
                    
                    <!--  -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 
        const networkData = {
            nodes: """ + str(nodes).replace("'", '"') + """,
            links: """ + str(links).replace("'", '"') + """
        };
        
        // ECharts
        const chart = echarts.init(document.getElementById('network-chart'));
        let currentFilter = 'all';
        
        function createNetworkOption(filterType = 'all') {
            let filteredNodes = [...networkData.nodes];
            let filteredLinks = [...networkData.links];
            
            // 
            if (filterType === 'inter-decade') {
                const interDecadeAuthors = new Set();
                filteredLinks = networkData.links.filter(link => link.type === 'inter-decade');
                
                if (filteredLinks.length > 0) {
                    filteredLinks.forEach(link => {
                        interDecadeAuthors.add(link.source);
                        interDecadeAuthors.add(link.target);
                    });
                    filteredNodes = networkData.nodes.filter(node => interDecadeAuthors.has(node.id));
                } else {
                    // 
                    filteredNodes = [...networkData.nodes];
                    filteredLinks = [];
                }
            } else if (filterType === 'batty-focus') {
                const battyConnected = new Set();
                // Batty
                networkData.nodes.forEach(node => {
                    if (node.id.includes('Batty')) {
                        battyConnected.add(node.id);
                    }
                });
                
                filteredLinks = networkData.links.filter(link => 
                    link.source.includes('Batty') || link.target.includes('Batty')
                );
                
                if (filteredLinks.length > 0) {
                    filteredLinks.forEach(link => {
                        battyConnected.add(link.source);
                        battyConnected.add(link.target);
                    });
                }
                
                filteredNodes = networkData.nodes.filter(node => battyConnected.has(node.id));
            }
            
            return {
                title: {
                    text: '',
                    left: 'center',
                    textStyle: { color: '#fff' }
                },
                tooltip: {
                    trigger: 'item',
                    formatter: function (params) {
                        if (params.dataType === 'node') {
                            return `
                                <div style="padding: 12px; font-family: 'Segoe UI', sans-serif;">
                                    <strong style="color: #2c3e50; font-size: 16px;">${params.data.full_name || params.data.name}</strong><br/>
                                    <div style="margin: 8px 0; padding: 4px 8px; background: ${params.data.color}; color: white; border-radius: 4px; display: inline-block;">
                                        ${params.data.first_decade}
                                    </div><br/>
                                    <span style="color: #7f8c8d;">Papers: </span><strong style="color: #3498db;">${params.data.papers || 'N/A'}</strong><br/>
                                    <span style="color: #7f8c8d;">Collaborations: </span><strong style="color: #3498db;">${params.data.degree}</strong>
                                </div>
                            `;
                        } else if (params.dataType === 'edge') {
                            return `
                                <div style="padding: 12px; font-family: 'Segoe UI', sans-serif;">
                                    <strong style="color: #2c3e50; font-size: 14px;">Collaboration Link</strong><br/>
                                    <div style="margin: 6px 0; color: #34495e;">
                                        ${params.data.source}  ${params.data.target}
                                    </div>
                                    <span style="color: #7f8c8d;">Period: </span><strong style="color: #3498db;">${params.data.decades}</strong><br/>
                                    <span style="color: #7f8c8d;">Strength: </span><strong style="color: #3498db;">${params.data.weight}</strong><br/>
                                    <div style="margin-top: 4px; padding: 2px 6px; background: ${params.data.type === 'inter-decade' ? '#9260f0' : '#95a5a6'}; color: white; border-radius: 3px; display: inline-block; font-size: 12px;">
                                        ${params.data.type === 'inter-decade' ? 'Cross-Decade' : 'Same Decade'}
                                    </div>
                                </div>
                            `;
                        }
                    },
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderColor: '#3498db',
                    borderWidth: 1,
                    textStyle: { color: '#2c3e50' }
                },
                series: [{
                    name: 'Academic Network',
                    type: 'graph',
                    layout: 'force',
                    data: filteredNodes.map(node => ({
                        ...node,
                        itemStyle: {
                            color: node.color,
                            borderColor: '#fff',
                            borderWidth: 1,
                            shadowBlur: 6,
                            shadowColor: 'rgba(0, 0, 0, 0.2)'
                        },
                        label: {
                            show: true,  //  
                            formatter: '{b}', //   name 
                            position: 'right',
                            distance: 15,  //  
                            fontSize: node.id.includes('Batty') ? 16 : 12,  //  
                            fontWeight: node.id.includes('Batty') ? 'bold' : 'normal',
                            color: '#2c3e50',
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',  //  
                            padding: [4, 8],  //  
                            borderRadius: 6,  //  
                            borderColor: node.color,
                            borderWidth: 1,
                            shadowBlur: 4,    //  
                            shadowColor: 'rgba(0, 0, 0, 0.1)',
                            shadowOffsetX: 1,
                            shadowOffsetY: 1
                        },
                        emphasis: {
                            focus: 'adjacency',
                            itemStyle: {
                                borderWidth: 4,
                                shadowBlur: 15,
                                shadowColor: node.color
                            },
                            label: {
                                show: true,
                                fontSize: 13,
                                fontWeight: 'bold'
                            }
                        }
                    })),
                    links: filteredLinks.map(link => ({
                        ...link,
                        lineStyle: {
                            color: link.color || link.lineStyle.color,  //  
                            width: Math.max(2, link.weight * 3),        //  
                            opacity: link.type === 'inter-decade' ? 0.9 : 0.7,
                            curveness: link.type === 'inter-decade' ? 0.25 : 0.1
                        }
                    })),
                    force: {
                        repulsion: 15000,       //  
                        gravity: 0.02,          //  
                        edgeLength: [400, 1000], //  
                        friction: 0.2,          //  
                        layoutAnimation: true,
                        preventOverlap: true,   //  
                        nodeStrength: 2000,     //  
                        coolingFactor: 0.99,    //  
                        iterations: 300,        //  
                        initialPositions: function(node) {  //  
                            return {
                                x: Math.random() * 2000 - 1000,
                                y: Math.random() * 2000 - 1000
                            };
                        }
                    },
                    roam: true,
                    focusNodeAdjacency: true,
                    draggable: true,
                    symbol: 'circle',
                    symbolSize: function(value, params) {
                        return params.data.symbolSize;
                    },
                    labelLayout: {
                        hideOverlap: false,     //  
                        moveOverlap: 'shiftXY', //  
                        draggable: true,        //  
                        labelLineHeight: 20,    //  
                        align: 'left'           //  
                    },
                    scaleLimit: {
                        min: 0.05,  //  
                        max: 15     //  
                    },
                    emphasis: {
                        focus: 'adjacency',
                        lineStyle: {
                            width: 6,
                            opacity: 1
                        }
                    }
                }]
            };
        }
        
        // 
        function showAllNodes() {
            currentFilter = 'all';
            chart.setOption(createNetworkOption('all'));
            updateActiveButton(0);
        }
        
        function showInterDecadeOnly() {
            currentFilter = 'inter-decade';
            chart.setOption(createNetworkOption('inter-decade'));
            updateActiveButton(1);
        }
        
        function focusBatty() {
            currentFilter = 'batty-focus';
            chart.setOption(createNetworkOption('batty-focus'));
            updateActiveButton(2);
        }
        
        function resetZoom() {
            chart.dispatchAction({
                type: 'restore'
            });
        }
        
        function saveAsImage() {
            //  PNG
            const url = chart.getDataURL({
                type: 'png',
                pixelRatio: 2,  // 
                backgroundColor: '#ffffff'
            });
            
            // 
            const link = document.createElement('a');
            link.href = url;
            link.download = `Community-""" + str(community_id) + """-Evolution_Network.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        function updateActiveButton(index) {
            document.querySelectorAll('.control-button').forEach((btn, i) => {
                btn.classList.toggle('active', i === index);
            });
        }
        
        // 
        chart.setOption(createNetworkOption());
        
        // 
        window.addEventListener('resize', function() {
            chart.resize();
        });
        
        // 
        chart.on('dblclick', function() {
            resetZoom();
        });
    </script>
</body>
</html>"""
        
        return html_content
    
    def _draw_author_label(self, ax, x: float, y: float, member: Dict, color: str):
        """
        
        """
        name = member['name']
        degree = member.get('degree', 0)
        
        # 
        importance = min(1.0, degree / 8.0) if degree > 0 else 0.5
        alpha = 0.6 + 0.4 * importance
        
        # 
        text_width = len(name) * 0.12 + 0.6
        rect = plt.Rectangle((x - 0.1, y - 0.25), text_width, 0.5,
                           facecolor=color, alpha=alpha, 
                           edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        
        # 
        text_color = 'white' if alpha > 0.7 else 'black'
        ax.text(x, y, name, fontsize=11, ha='left', va='center',
               color=text_color, fontweight='normal')
    
    def _draw_timeline(self, ax, start_x: float, end_x: float, y: float, decades: List[str]):
        """
        
        """
        # 
        ax.plot([start_x, end_x], [y, y], 'k-', linewidth=1, alpha=0.5)
        
        # 
        for i, decade in enumerate(decades):
            x_pos = start_x + (end_x - start_x) * i / (len(decades) - 1)
            ax.plot([x_pos, x_pos], [y - 0.1, y + 0.1], 'k-', linewidth=1, alpha=0.5)
            ax.text(x_pos, y + 0.4, decade, ha='center', va='bottom', 
                   fontsize=10, alpha=0.8)
    
    def _analyze_intergenerational_connections(self, comm_id: int, decades: List[str], results: Dict) -> Dict:
        """
        
        
        Args:
            comm_id: ID
            decades: 
            results: 
            
        Returns:
            Dict: 
        """
        print(f"    Analyzing intergenerational connections for Community {comm_id}...")
        
        # 
        decade_leaders = {}
        all_members = []
        
        for decade in decades:
            if decade in results['decade_analysis']:
                comm_data = results['decade_analysis'][decade]['communities'].get(comm_id, {})
                leaders = comm_data.get('leaders', [])
                
                decade_members = []
                for leader_info in leaders[:5]:  # 5
                    if leader_info:
                        member_data = {
                            'name': leader_info[0],
                            'degree': leader_info[1] if len(leader_info) > 1 else 0,
                            'papers': leader_info[2] if len(leader_info) > 2 else 0,
                            'decade': decade,
                            'decade_index': decades.index(decade)
                        }
                        decade_members.append(member_data)
                        all_members.append(member_data)
                
                if decade_members:
                    decade_leaders[decade] = decade_members
        
        #  
        intergenerational_connections = []
        
        for i, decade1 in enumerate(decades[:-1]):  # 
            for j, decade2 in enumerate(decades[i+1:], start=i+1):  # 
                
                leaders1 = decade_leaders.get(decade1, [])
                leaders2 = decade_leaders.get(decade2, [])
                
                for leader1 in leaders1:
                    for leader2 in leaders2:
                        # 
                        collaboration_strength = self._calculate_collaboration_strength(
                            leader1['name'], leader2['name']
                        )
                        
                        if collaboration_strength > 0.1:  # 
                            connection = {
                                'mentor': leader1['name'],
                                'mentee': leader2['name'],
                                'mentor_decade': decade1,
                                'mentee_decade': decade2,
                                'decade_gap': j - i,
                                'strength': collaboration_strength,
                                'mentor_data': leader1,
                                'mentee_data': leader2
                            }
                            intergenerational_connections.append(connection)
        
        # 
        intergenerational_connections.sort(key=lambda x: x['strength'], reverse=True)
        
        #  
        lineage_trees = self._build_lineage_trees(intergenerational_connections, all_members)
        
        return {
            'decade_leaders': decade_leaders,
            'all_members': all_members,
            'connections': intergenerational_connections,
            'lineage_trees': lineage_trees,
            'community_id': comm_id
        }
    
    def _calculate_collaboration_strength(self, author1: str, author2: str) -> float:
        """
        
        
        Args:
            author1: 1
            author2: 2
            
        Returns:
            float:  (0-1)
        """
        if not self.network.has_edge(author1, author2):
            return 0.0
        
        edge_data = self.network[author1][author2]
        weight = edge_data.get('weight', 0)
        count = edge_data.get('count', 0)
        
        # 
        strength = min(1.0, (weight * 0.7 + count * 0.3))
        return strength
    
    def _build_lineage_trees(self, connections: List[Dict], all_members: List[Dict]) -> List[Dict]:
        """
        
        
        Args:
            connections: 
            all_members: 
            
        Returns:
            List[Dict]: 
        """
        if not connections:
            # ""
            trees = []
            for member in all_members:
                trees.append({
                    'root': member,
                    'members': [member],
                    'connections': [],
                    'tree_type': 'isolated'
                })
            return trees
        
        # 
        import networkx as nx
        lineage_graph = nx.DiGraph()
        
        # 
        for member in all_members:
            lineage_graph.add_node(member['name'], data=member)
        
        # 
        for conn in connections:
            lineage_graph.add_edge(conn['mentor'], conn['mentee'], connection_data=conn)
        
        # 
        trees = []
        undirected_graph = lineage_graph.to_undirected()
        
        for component in nx.connected_components(undirected_graph):
            if len(component) == 1:
                # 
                member_name = list(component)[0]
                member_data = lineage_graph.nodes[member_name]['data']
                tree = {
                    'root': member_data,
                    'members': [member_data],
                    'connections': [],
                    'tree_type': 'isolated'
                }
            else:
                # 
                subgraph = lineage_graph.subgraph(component)
                
                # 0
                roots = [n for n in subgraph.nodes() if subgraph.in_degree(n) == 0]
                root_node = roots[0] if roots else list(component)[0]
                
                tree_members = []
                tree_connections = []
                
                for node in component:
                    tree_members.append(lineage_graph.nodes[node]['data'])
                
                for edge in subgraph.edges(data=True):
                    tree_connections.append(edge[2]['connection_data'])
                
                tree = {
                    'root': lineage_graph.nodes[root_node]['data'],
                    'members': sorted(tree_members, key=lambda x: x['decade_index']),
                    'connections': tree_connections,
                    'tree_type': 'lineage'
                }
            
            trees.append(tree)
        
        # 
        trees.sort(key=lambda x: (x['tree_type'] != 'lineage', -len(x['members'])))
        
        return trees
    
    
    def build_community_network(self, communities_result, plot: bool = True):
        """
        6: 
        
        Args:
            communities_result: 
            plot: 
        """
        print("\n" + "="*50)
        print(" 6: ")
        print("="*50)
        
        if communities_result is None:
            print(" ")
            return None
        
        partition = communities_result['partition']
        
        # 
        community_edges = {}
        community_weights = {}
        
        for u, v, data in self.largest_subgraph.edges(data=True):
            comm_u = partition[u]
            comm_v = partition[v]
            
            if comm_u != comm_v:  # 
                edge = tuple(sorted([comm_u, comm_v]))
                community_edges[edge] = community_edges.get(edge, 0) + 1
                community_weights[edge] = community_weights.get(edge, 0) + data.get('weight', 1)
        
        # 
        community_network = nx.Graph()
        
        # 
        community_sizes = communities_result['community_sizes']
        for community_id, size in community_sizes.items():
            community_network.add_node(community_id, size=size)
        
        # 
        for edge, count in community_edges.items():
            comm1, comm2 = edge
            weight = community_weights[edge]
            community_network.add_edge(comm1, comm2, 
                                     edge_count=count, 
                                     total_weight=weight,
                                     avg_weight=weight/count)
        
        print(f"Community network statistics:")
        print(f"  Community nodes: {community_network.number_of_nodes()}")
        print(f"  Inter-community links: {community_network.number_of_edges()}")
        print(f"  Community network density: {nx.density(community_network):.4f}")
        
        return {
            'community_network': community_network,
            'community_edges': community_edges,
            'community_weights': community_weights
        }
    
    def run_complete_analysis(self, use_weighted: bool = True):
        """
        
        
        Args:
            use_weighted: 
        """
        print(" Starting comprehensive network structural analysis")
        print("="*60)
        
        results = {}
        
        # 1: 
        results['degree_distribution'] = self.analyze_degree_distribution(use_weighted=use_weighted)
        
        # 2: 
        results['degree_correlation'] = self.analyze_degree_correlation()
        
        # 3: 
        results['clustering'] = self.analyze_clustering_coefficient()
        
        # 4: 
        results['small_world'] = self.analyze_small_world()
        
        # 5: 
        results['communities'] = self.detect_communities(random_seed=42, standardize_ids=True)
        
        # 6: 
        if results['communities'] is not None:
            results['community_network'] = self.build_community_network(results['communities'])
        
        print("\n" + "="*60)
        print(" Comprehensive analysis completed!")
        print("="*60)
        
        return results
    
    def get_last_name(self, full_name: str) -> str:
        """ (Last Name)"""
        if ',' in full_name:
            #  "Last, First" 
            return full_name.split(',')[0].strip()
        else:
            #  "First Last" 
            parts = full_name.split()
            return parts[0]

    
    