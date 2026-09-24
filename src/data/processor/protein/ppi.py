"""
PPI (Protein-Protein Interaction) network loader for KEPPNet.

Required public data files (download instructions in docs/data_preparation.md):
  data/protein/9606.protein.links.full.v12.0.txt  — STRING v12 full network
  data/protein/9606.protein.info.v12.0.txt         — STRING v12 protein names

Optional KG-augmentation file (cancer-type-specific, not publicly distributed):
  data/protein/{cancer_type}_id2id_2hop.csv        — 2-hop KG protein relations
  If absent, the standard STRING PPI network is used without KG augmentation.

Optional hub-gene ranking file (not required for core functionality):
  data/genes/expressed_genes_and_cancer_genes.csv  — gene ranking for hub selection
  If absent, hub selection falls back to degree-only ranking.
"""

import os
import re
import warnings
import networkx as nx
import pandas as pd
from os.path import join, exists

ppi_base_dir  = os.path.join(os.getcwd(), 'data/protein')
gene_base_dir = os.path.join(os.getcwd(), 'data/genes')

# STRING v12 filenames (public, downloadable)
relations_file_name = '9606.protein.links.full.v12.0.txt'
protein_name_file   = '9606.protein.info.v12.0.txt'

# Optional gene-ranking file for hub selection (not required)
GENE_RANK_FILE = 'expressed_genes_and_cancer_genes.csv'


# ── graph utilities ───────────────────────────────────────────────────────────

def add_edges(G, node, n_levels):
    edges, source = [], node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edges.append((source, target))
        source = target
    G.add_edges_from(edges)
    return G


def complete_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    terminal_nodes = [n for n, d in nx.bfs_tree(sub_graph, 'root').out_degree() if d == 0]
    terminal_nodes = sorted(terminal_nodes)
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)
    return sub_graph


def get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))
    return sorted(list(nodes))


def get_top_node(net, id2name, name2id):
    """
    Select hub root nodes by degree + optional gene-rank file.
    Falls back to degree-only if the gene-rank file is not available.
    """
    nodes = list(net.nodes())
    top_num = max(len(nodes) // 100 * 3, 2)
    valid_node_name = [id2name[n] for n in nodes if n in id2name]

    degree_dict = dict(net.degree())
    sorted_nodes_by_degree = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)

    gene_rank_path = join(gene_base_dir, GENE_RANK_FILE)
    if exists(gene_rank_path):
        df_gene = pd.read_csv(gene_rank_path)
        gene_list = df_gene['genes'].to_list()
        sorted_nodes_by_contribution = [name2id[gene] for gene in gene_list
                                        if gene in valid_node_name and gene in name2id]
        candidates = sorted(list(
            set(sorted_nodes_by_degree[:top_num]) &
            set(sorted_nodes_by_contribution[:top_num])
        ))
        if candidates:
            return candidates
        # intersection empty — fall through to degree-only
        warnings.warn(
            f"Gene rank file found but intersection with degree-top nodes is empty. "
            "Falling back to degree-only hub selection."
        )
    else:
        warnings.warn(
            f"Optional gene rank file not found: {gene_rank_path}. "
            "Using degree-only hub selection (standard behaviour for open-source release)."
        )

    # Degree-only fallback
    return sorted(sorted_nodes_by_degree[:top_num])


def get_layers_from_net(net, n_levels):
    layers = []
    dis2nodes = {}
    for i in range(n_levels + 1):
        dis2nodes[i] = get_nodes_at_level(net, i)
    for i in range(n_levels):
        layer_dict = {}
        for n in dis2nodes[i]:
            n_name = re.sub('_copy.*', '', n)
            layer_dict[n_name] = [
                re.sub('_copy.*', '', nex)
                for nex in net.neighbors(n)
                if nex in dis2nodes[i + 1]
            ]
        layers.append(layer_dict)
    return layers


# ── Protein class ─────────────────────────────────────────────────────────────

class Protein:
    """
    Loads STRING PPI data and optionally augments with a KG 2-hop file.

    The KG file (e.g. data/protein/breast_id2id_2hop.csv) is cancer-type-specific
    and not publicly distributed. If absent, the standard STRING network is used.
    """

    def __init__(self, genes, cancer_type: str = None):
        """
        Args:
            genes: iterable of gene symbols to filter the PPI network
            cancer_type: optional cancer type string used to locate the KG file
                         (e.g. 'breast', 'prostate'). If None or file absent,
                         KG augmentation is skipped.
        """
        self.cancer_type = cancer_type
        self.protein_name = self.load_names(genes)
        self.name2id = self.protein_name.set_index('name')['protein_id'].to_dict()
        self.id2name = self.protein_name.set_index('protein_id')['name'].to_dict()
        self.hierarchy = self.load_hierarchy()

    def load_names(self, genes):
        filename = join(ppi_base_dir, protein_name_file)
        if not exists(filename):
            raise FileNotFoundError(
                f"STRING protein info file not found: {filename}\n"
                "Download from https://string-db.org/cgi/download "
                "and place in data/protein/. See docs/data_preparation.md."
            )
        df = pd.read_csv(filename, sep='\t')
        df.rename(columns={'#string_protein_id': 'protein_id', 'preferred_name': 'name'},
                  inplace=True)
        df = df[df['name'].isin(genes)]
        df.reset_index(drop=True, inplace=True)
        return df

    def load_hierarchy(self):
        """
        Build PPI edge table from STRING links file.
        Optionally intersects with a KG 2-hop file if available.
        Returns a DataFrame with columns ['child', 'parent'].
        """
        cache_path = join(ppi_base_dir, 'selected_links.txt')

        # Always regenerate cache to avoid stale data across cancer types
        if exists(cache_path):
            os.remove(cache_path)

        links_file = join(ppi_base_dir, relations_file_name)
        if not exists(links_file):
            raise FileNotFoundError(
                f"STRING PPI links file not found: {links_file}\n"
                "Download from https://string-db.org/cgi/download "
                "and place in data/protein/. See docs/data_preparation.md."
            )

        df = pd.read_csv(links_file, sep=' ', low_memory=False)

        # Filter to genes of interest
        select_ids = set(self.id2name.keys())
        df = df[df['protein1'].isin(select_ids) & df['protein2'].isin(select_ids)]

        # Confidence threshold (top 30%)
        confidence = 0.7
        score_range = df['combined_score'].max() - df['combined_score'].min()
        threshold = df['combined_score'].min() + score_range * confidence
        df = df[df['combined_score'] > threshold]

        # Deduplicate undirected edges
        df['_flag'] = df.apply(
            lambda r: r['protein1'] + r['protein2']
            if r['protein1'] < r['protein2']
            else r['protein2'] + r['protein1'],
            axis=1,
        )
        df = df[~df['_flag'].duplicated(keep='first')].drop('_flag', axis=1)
        df.to_csv(cache_path, index=False)

        df = df[['protein1', 'protein2']].rename(
            columns={'protein1': 'child', 'protein2': 'parent'}
        ).reset_index(drop=True)

        # ── Optional KG augmentation ──────────────────────────────────────────
        # The KG 2-hop file is cancer-type-specific and not publicly distributed.
        # If absent, the standard STRING network is used without modification.
        kg_path = None
        if self.cancer_type:
            kg_path = join(ppi_base_dir, f'{self.cancer_type}_id2id_2hop.csv')

        if kg_path and exists(kg_path):
            print(f"[PPI] Loading KG augmentation from {kg_path}")
            kg_df = pd.read_csv(kg_path, low_memory=False)

            # Keep only KG edges that are consistent with the STRING network
            ppi_pairs = set(tuple(x) for x in df[['child', 'parent']].values)
            ppi_pairs |= {(b, a) for a, b in ppi_pairs}
            mask = kg_df.apply(
                lambda r: (r['child'], r['parent']) in ppi_pairs, axis=1
            )
            filtered_kg = kg_df[mask]
            print(f"[PPI] KG edges after filtering: {len(filtered_kg)}")
            return filtered_kg
        else:
            if self.cancer_type:
                warnings.warn(
                    f"KG augmentation file not found: {kg_path}\n"
                    "Using standard STRING PPI network without KG augmentation.\n"
                    "This is expected for the open-source release. "
                    "Results may differ slightly from the paper."
                )
            return df


# ── ProteinNetwork class ──────────────────────────────────────────────────────

class ProteinNetwork:

    def __init__(self, genes, cancer_type: str = None):
        self.protein = Protein(genes, cancer_type=cancer_type)
        self.netx = self.get_networkx()

    def get_roots(self):
        return get_nodes_at_level(self.netx, distance=1)

    def get_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.protein.hierarchy
        net = nx.from_pandas_edgelist(hierarchy, 'child', 'parent')
        components = list(nx.connected_components(net))
        largest_component = sorted(max(components, key=len))
        pd_largest = hierarchy[
            hierarchy['child'].isin(largest_component) &
            hierarchy['parent'].isin(largest_component)
        ]
        net = nx.from_pandas_edgelist(pd_largest, 'child', 'parent')
        net.name = 'ppi'
        roots = get_top_node(net, self.protein.id2name, self.protein.name2id)
        net.add_edges_from([('root', n) for n in roots])
        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):
        return nx.bfs_tree(self.netx, 'root')

    def get_completed_network(self, n_levels):
        return complete_network(self.netx, n_leveles=n_levels)

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        return complete_network(G, n_leveles=n_levels)

    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        terminal_nodes = get_nodes_at_level(net, n_levels)
        filter_gene = set(self.protein.name2id.values())

        layer_dict = {}
        missing = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            neig_gene = sorted(set(self.netx.adj[pathway_name].keys()))
            genes = [self.protein.id2name[g] for g in neig_gene if g in filter_gene]
            if not genes:
                missing.append(pathway_name)
            layer_dict[pathway_name] = genes

        layers.append(layer_dict)
        return layers
