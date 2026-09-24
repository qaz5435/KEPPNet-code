import os
import re

from os.path import join
from copy import deepcopy

import networkx as nx
import pandas as pd

from src.data.processor.pathways.gmt_reader import GMT
from src.data.processor.pathways.reactome import Reactome
from src.data.processor.protein.ppi import Protein

# reactome_base_dir = REACTOM_PATHWAY_PATH
# work_dir = os.getcwd()
# ppi_base_dir = os.path.join(work_dir, 'data/protein')
# gene_base_dir = os.path.join(work_dir, 'data/genes')
# relations_file_name = '9606.protein.links.full.v12.0.txt'
# protein_name = '9606.protein.info.v12.0.txt'
# gene_file_name = 'breast_expressed_genes_and_cancer_genes_10000.csv'

# reactome_base_dir = os.path.join(work_dir, "data/pathways/Reactome")
# relations_file_name = 'ReactomePathwaysRelation.txt'
# pathway_names = 'ReactomePathways.txt'
# pathway_genes = 'ReactomePathways.gmt'


def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G

def complete_pathway_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles) #get subgraph of neighbors centered at node root within 5 radius. 
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph

def complete_ppi_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles) #get subgraph of neighbors centered at node root within 5 radius. 
    # terminal_nodes = [n for n, d in sub_graph.degree() if d == 1]
    # terminal_nodes = [n for n, d in sub_graph.degree() if d < 5] #8
    terminal_nodes = [n for n, d in nx.bfs_tree(sub_graph, 'root').out_degree() if d == 0]
    distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph

def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))

    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)

def get_top_node(net, id2name, name2id):
    # return top node based on special rule.
    top_num = max(len(net.node) // 100 * 3, 2)
    valid_node_name = [id2name[id] for id in net.node]

    # degree
    degree_dict = dict(net.degree())
    sorted_nodes_by_degree = sorted(degree_dict, key=lambda x:degree_dict[x], reverse=True)
    
    # betweenness centrality
    betweenness_dict = nx.approximate_current_flow_betweenness_centrality(net)
    sorted_nodes_by_betweenness_centrality = sorted(betweenness_dict, key=lambda x: betweenness_dict[x], reverse=True)

    # contribution from genecard
    df_gene = pd.read_csv(join(gene_base_dir, gene_file_name))
    gene_list = df_gene['genes'].to_list()
    sorted_nodes_by_contribution = [name2id[gene] for gene in gene_list if gene in valid_node_name]

    return list(
        set(sorted_nodes_by_degree[:top_num]) &
        set(sorted_nodes_by_betweenness_centrality[:top_num]) &
        set(sorted_nodes_by_contribution[:top_num])
    )

def get_layers_from_net(net, n_levels):
    layers = []
    dis2nodes = {}
    for i in range(n_levels):
        dis2nodes[i] = get_nodes_at_level(net, i)
    
    # for i in range(n_levels + 1):
    #     dis2nodes[i] = []
    # for node, distance in nx.single_source_shortest_path_length(net, 'root').items():
    #     dis2nodes[distance].append(node)
    
    for i in range(n_levels):
        dict = {}
        for n in dis2nodes[i]:
            n_name = re.sub('_copy.*', '', n)
            # 修复：匹配任意位数的数字，如 -1.5, -10.5, -100.5 等
            n_name = re.sub(r"-\d+\.\d+$", "", n_name)
            next = net.successors(n)
            # dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
            dict[n_name] = []
            for nex in next:
                normal_nex = re.sub('_copy.*', '', nex)
                # 修复：匹配任意位数的数字
                normal_nex = re.sub(r"-\d+\.\d+$", "", normal_nex)
                dict[n_name].append(normal_nex)
        layers.append(dict)
    return layers


class ReactomeProteinNetwork():

    def __init__(self, genes, cancer_type: str = None):
        self.reactome = Reactome()
        self.protein = Protein(genes, cancer_type=cancer_type)
        self.netx = self.get_reactome_networkx()

    def insert_protein_between_pathways(self, net):
        # iter all edges
        pathway_genes = deepcopy(self.reactome.pathway_genes)
        pathway_genes['gene'].replace(self.protein.name2id, inplace=True)
        pathway_genes = pathway_genes[pathway_genes['gene'].str.startswith('9606')]
        suffix = r'_copy\d+'

        # dis2nodes = {}
        # for i in range(3 + 1):
        #     dis2nodes[i] = 0
        # for node, distance in nx.single_source_shortest_path_length(net, 'root').items():
        #     dis2nodes[distance] += 1
        # print(dis2nodes)

        add_edges_list = []
        remove_edges_list = []
        for edge in list(net.edges):
            
            start_pathway_s = edge[0]
            end_pathway_s = edge[1]
            if start_pathway_s == 'root':
                continue

            # remove suffix _copy*
            # start_pos = len(nx.shortest_path(net, source='root', target=start_pathway_s))
            end_pos = len(nx.shortest_path(net, source='root', target=end_pathway_s))
            start_pathway = re.sub(suffix, '', edge[0])
            end_pathway = re.sub(suffix, '', edge[1])

            # get related protein id with start pathway id
            start_protein = set(pathway_genes[
                pathway_genes['group'] == start_pathway]['gene'])

            # get related protein id with end pathway id
            end_protein = set(pathway_genes[
                pathway_genes['group'] == end_pathway]['gene'])

            # get intersection between protein id
            interset = start_protein.intersection(end_protein)

            # if len(interset) == 0: # exist ? weather remove previous edge?
            #     print("test")

            # # insert edges (start_pathway -> intersection protein id)
            # net.add_edges_from(
            #     [(start_pathway_s, node) for node in interset]
            # )

            # # insert edges (intersection protein id -> end_pathway)

            # net.add_edges_from(
            #     [(node, end_pathway_s) for node in interset]
            # )

            # net.remove_edge(start_pathway_s, end_pathway_s)
            for node in interset:
                add_edges_list.append((start_pathway_s, node + '-' + str(end_pos - 0.5)))
                add_edges_list.append((node + '-' + str(end_pos - 0.5), end_pathway_s))
            
            remove_edges_list.append((start_pathway_s, end_pathway_s))

        net.add_edges_from(add_edges_list)
        net.remove_edges_from(remove_edges_list)
        # dis2nodes = {}
        # for i in range(5 + 1):
        #     dis2nodes[i] = 0
        # for node, distance in nx.single_source_shortest_path_length(net, 'root').items():
        #     dis2nodes[distance] += 1
        # print(dis2nodes)

        return net
        

    def get_roots(self):

        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
        net.name = 'reactome'

        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    # get a undirected graph representation of the ppi hierarchy
    def get_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.protein.hierarchy
        net = nx.from_pandas_edgelist(hierarchy, 'child', 'parent')
        components = list(nx.connected_components(net))
        largest_component = max(components, key=len)
        largest_subgraph = net.subgraph(largest_component).copy()
        net = largest_subgraph
        net.name = 'ppi'
        roots = get_top_node(net, self.protein.id2name, self.protein.name2id)
        # roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):

        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')

        return G

    def get_completed_network(self, n_levels):
        G = complete_pathway_network(self.netx, n_leveles=n_levels)
        return G

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = self.get_completed_network(G, n_leveles=n_levels)
        return G
    
    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            net = self.insert_protein_between_pathways(net)
            layers = get_layers_from_net(net, n_levels + n_levels - 1)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        # get the last layer (genes level)
        # terminal_nodes = [n for n, d in net.degree() if d < 5]  # set of terminal pathways
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]
        # terminal_nodes = list(layers[n_levels - 1].keys())

        # we need to find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            genes = genes_df[genes_df['group'] == pathway_name]['gene'].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes

        layers.append(dict)
        return layers
