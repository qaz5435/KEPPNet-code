"""
知识增强版的Reactome-Protein混合网络
支持基于特定癌症数据集的网络稀疏化
"""

import os
import re
import json
import networkx as nx
import pandas as pd
from os.path import join
from copy import deepcopy

from src.data.processor.pathways.reactome import Reactome
from src.data.processor.protein.ppi import Protein


def add_edges(G, node, n_levels):
    """添加虚拟节点链以补全网络"""
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
    """补全Reactome通路网络，确保所有路径长度一致"""
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)
    
    return sub_graph


def get_nodes_at_level(net, distance):
    """获取距离root恰好为distance的所有节点"""
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    
    if distance >= 1:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))
    
    return list(nodes)


def get_layers_from_net(net, n_levels):
    """从网络中提取层级结构"""
    layers = []
    dis2nodes = {}
    
    for i in range(n_levels):
        dis2nodes[i] = get_nodes_at_level(net, i)
    
    for i in range(n_levels):
        layer_dict = {}
        for node in dis2nodes[i]:
            # 移除所有后缀
            node_name = re.sub('_copy.*', '', node)
            # 修复：匹配任意位数的数字，如 -1.5, -10.5, -100.5 等
            node_name = re.sub(r"-\d+\.\d+$", "", node_name)
            
            # 获取后继节点
            successors = net.successors(node)
            layer_dict[node_name] = []
            
            for succ in successors:
                normal_succ = re.sub('_copy.*', '', succ)
                # 修复：匹配任意位数的数字
                normal_succ = re.sub(r"-\d+\.\d+$", "", normal_succ)
                layer_dict[node_name].append(normal_succ)
        
        layers.append(layer_dict)
    
    return layers



class ReactomeProteinNetworkKGEnhanced:
    """
    知识增强版的Reactome-Protein混合网络
    
    特点:
    1. 基于特定癌症数据集筛选基因
    2. 根据基因筛选相关的通路
    3. 根据基因筛选相关的蛋白质
    4. 使用2-hop知识图谱增强连接
    5. 生成更稀疏、更针对性的网络
    """
    
    def __init__(self, genes, cancer_type='breast', use_kg=True, use_processed_data=True,
                 min_shared_proteins=2, min_shared_ratio=0.05,
                 use_dynamic_threshold=True, filter_hub_proteins=True,
                 hub_threshold=50, max_hub_proteins=3,
                 max_mediators_per_edge=10):
        """
        Args:
            genes: 基因列表（已经根据癌症类型筛选过的）
            cancer_type: 癌症类型 ('breast', 'prostate', 'lung'等)
            use_kg: 是否使用知识图谱增强
            use_processed_data: 是否使用预处理的Reactome数据
            min_shared_proteins: 最小共享蛋白数（基础阈值）
            min_shared_ratio: 最小共享比例（基础阈值）
            use_dynamic_threshold: 是否使用动态层级阈值
            filter_hub_proteins: 是否过滤枢纽蛋白
            hub_threshold: 枢纽蛋白阈值（参与通路数）
            max_hub_proteins: 当非枢纽蛋白不足时，最多添加的枢纽蛋白数
            max_mediators_per_edge: 每个通路对最多保留的介导蛋白数
        """
        self.genes = genes
        self.cancer_type = cancer_type
        self.use_kg = use_kg
        
        # 阈值策略参数
        self.min_shared_proteins = min_shared_proteins
        self.min_shared_ratio = min_shared_ratio
        self.use_dynamic_threshold = use_dynamic_threshold
        self.filter_hub_proteins = filter_hub_proteins
        self.hub_threshold = hub_threshold
        self.max_hub_proteins = max_hub_proteins
        self.max_mediators_per_edge = max_mediators_per_edge
        
        print(f"\n{'='*70}")
        print(f"Initializing KG-Enhanced Hybrid Network")
        print(f"  Cancer type: {cancer_type}")
        print(f"  Input genes: {len(genes)}")
        print(f"  Use KG: {use_kg}")
        print(f"{'='*70}\n")
        
        # 加载Reactome数据
        self.reactome = Reactome()
        
        # 加载通路-蛋白质映射
        if use_processed_data:
            self.pathway_protein_dict = self.load_processed_mapping()
        else:
            self.pathway_protein_dict = self.build_mapping_from_genes()

        # Load STRING evidence over the proteins that occur in retained
        # pathway annotations.  Protein() applies the same confidence filter as
        # the rest of the repository and returns identifiers from STRING v12.
        self.ppi_graph, self.ppi_id2name = self.load_ppi_graph()
        
        # 如果使用KG，加载KG增强数据
        if use_kg:
            self.kg_gene_pathway = self.load_kg_gene_pathway()
            self.kg_protein_protein = self.load_kg_protein_protein()
            if self.kg_gene_pathway is None or self.kg_protein_protein is None:
                raise FileNotFoundError(
                    "The main KG-enhanced configuration requires both cohort-specific "
                    "two-hop relation tables. See docs/data_preparation.md. Use the "
                    "corresponding *-no-kg configuration for the direct-evidence ablation."
                )
        else:
            self.kg_gene_pathway = None
            self.kg_protein_protein = None
        
        # 根据基因筛选相关通路和蛋白质
        self.filtered_pathways = self.filter_pathways_by_genes()
        self.filtered_proteins = self.filter_proteins_by_genes()
        
        # 构建网络
        self.netx = self.get_reactome_networkx()
        
        # 识别枢纽蛋白（如果需要）
        if self.filter_hub_proteins:
            self.hub_proteins = self.identify_hub_proteins()
        else:
            self.hub_proteins = set()
    
    def load_processed_mapping(self):
        """加载预处理的通路-蛋白质映射"""
        processed_dir = 'data/pathway_protein/processed'
        mapping_file = join(processed_dir, 'pathway_to_proteins.json')
        
        if not os.path.exists(mapping_file):
            print(f"Warning: Processed mapping file not found: {mapping_file}")
            return self.build_mapping_from_genes()
        
        print(f"Loading processed pathway-protein mapping...")
        with open(mapping_file, 'r') as f:
            pathway_protein_dict = json.load(f)
        
        print(f"  Loaded {len(pathway_protein_dict)} pathways")
        return pathway_protein_dict
    
    def build_mapping_from_genes(self):
        """从基因构建映射（备用方案）"""
        print("Building pathway-protein mapping from genes...")
        pathway_genes = deepcopy(self.reactome.pathway_genes)
        
        pathway_protein_dict = {}
        for pathway_id in pathway_genes['group'].unique():
            proteins = pathway_genes[pathway_genes['group'] == pathway_id]['gene'].tolist()
            pathway_protein_dict[pathway_id] = proteins
        
        return pathway_protein_dict

    def load_ppi_graph(self):
        """Load a symbol-level STRING graph for cross-pathway evidence.

        Reactome supplies pathway membership, while STRING supplies the
        interaction evidence used by :meth:`find_ppi_bridge_proteins`.  Keeping
        those roles separate mirrors Eq. (4) and Algorithm 1 of the manuscript.
        """
        pathway_proteins = sorted({
            protein
            for proteins in self.pathway_protein_dict.values()
            for protein in proteins
            if isinstance(protein, str) and protein
        })
        if not pathway_proteins:
            raise ValueError("No pathway-associated proteins were available")

        protein_resource = Protein(pathway_proteins, cancer_type=None)
        graph = nx.Graph()
        for row in protein_resource.hierarchy.itertuples(index=False):
            child = protein_resource.id2name.get(row.child)
            parent = protein_resource.id2name.get(row.parent)
            if child and parent and child != parent:
                graph.add_edge(child, parent)

        if graph.number_of_edges() == 0:
            raise ValueError(
                "No STRING interactions remained after pathway-protein and "
                "confidence filtering"
            )
        print(
            f"Loaded STRING evidence: {graph.number_of_nodes()} proteins, "
            f"{graph.number_of_edges()} interactions"
        )
        return graph, protein_resource.id2name

    
    def load_kg_gene_pathway(self):
        """
        加载知识图谱：基因-通路2-hop关系
        格式: genes, pathways
        """
        kg_file = f'data/pathways/Reactome/kg/{self.cancer_type}/genes2pathways_2hop_v1.csv'
        
        if not os.path.exists(kg_file):
            print(f"Required KG file not found: {kg_file}")
            return None
        
        print(f"Loading KG gene-pathway mapping from {kg_file}")
        df = pd.read_csv(kg_file)
        df.columns = ['gene', 'pathway']
        
        # 只保留在输入基因列表中的基因
        df = df[df['gene'].isin(self.genes)]
        
        print(f"  KG entries: {len(df)}")
        print(f"  Unique genes: {df['gene'].nunique()}")
        print(f"  Unique pathways: {df['pathway'].nunique()}")
        
        return df
    
    def load_kg_protein_protein(self):
        """
        加载知识图谱：蛋白质-蛋白质2-hop关系
        格式: child, parent (蛋白质ID)
        """
        kg_file = f'data/protein/{self.cancer_type}_id2id_2hop.csv'
        
        if not os.path.exists(kg_file):
            print(f"Required KG file not found: {kg_file}")
            return None
        
        print(f"Loading KG protein-protein mapping from {kg_file}")
        df = pd.read_csv(kg_file)
        required = {'child', 'parent'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{kg_file} is missing columns: {sorted(missing)}")

        # Accept either STRING identifiers or gene/protein symbols in the
        # derived relation table and normalize to the symbol space used by the
        # pathway sets.
        for column in ('child', 'parent'):
            original = df[column].astype(str)
            df[column] = original.map(self.ppi_id2name).fillna(original)
        
        print(f"  KG protein-protein edges: {len(df)}")
        print(f"  Unique proteins: {pd.concat([df['child'], df['parent']]).nunique()}")
        
        return df
    
    def filter_pathways_by_genes(self):
        """
        根据输入基因筛选相关通路
        
        策略:
        1. 如果使用KG: 使用KG中的基因-通路映射
        2. 否则: 使用标准的pathway_genes映射
        """
        print("\nFiltering pathways by genes...")
        
        if self.use_kg and self.kg_gene_pathway is not None:
            # 使用KG增强的映射
            filtered_pathways = set(self.kg_gene_pathway['pathway'].unique())
            print(f"  Using KG: {len(filtered_pathways)} pathways")
        else:
            # 使用标准映射
            pathway_genes = self.reactome.pathway_genes
            # 找到包含输入基因的所有通路
            filtered_pathways = set(
                pathway_genes[pathway_genes['gene'].isin(self.genes)]['group'].unique()
            )
            print(f"  Using standard mapping: {len(filtered_pathways)} pathways")
        
        return filtered_pathways
    
    def filter_proteins_by_genes(self):
        """
        根据输入基因筛选相关蛋白质
        
        策略:
        1. 从pathway_protein_dict中提取与筛选后通路相关的蛋白质
        2. 如果使用KG: 额外添加KG中的蛋白质关系
        """
        print("\nFiltering proteins by genes...")
        
        filtered_proteins = set()
        
        # 从筛选后的通路中提取蛋白质
        for pathway_id in self.filtered_pathways:
            proteins = self.pathway_protein_dict.get(pathway_id, [])
            filtered_proteins.update(proteins)
        
        print(f"  Proteins from filtered pathways: {len(filtered_proteins)}")
        
        # 如果使用KG，添加额外的蛋白质
        if self.use_kg and self.kg_protein_protein is not None:
            kg_proteins = set(self.kg_protein_protein['child'].unique()) | \
                         set(self.kg_protein_protein['parent'].unique())
            
            # 只保留与当前蛋白质集合有交集的KG蛋白质
            # 这样可以避免引入太多不相关的蛋白质
            original_count = len(filtered_proteins)
            filtered_proteins.update(kg_proteins)
            print(f"  Added {len(filtered_proteins) - original_count} proteins from KG")
        
        print(f"  Total filtered proteins: {len(filtered_proteins)}")
        
        return filtered_proteins

    
    def identify_hub_proteins(self):
        """
        识别枢纽蛋白（参与过多通路的蛋白质）
        
        Returns:
            set: 枢纽蛋白集合
        """
        print(f"\nIdentifying hub proteins (threshold={self.hub_threshold})...")
        
        protein_pathway_count = {}
        
        # 统计每个蛋白质参与的通路数
        for pathway_id, proteins in self.pathway_protein_dict.items():
            if pathway_id not in self.filtered_pathways:
                continue
            
            for protein in proteins:
                if protein not in self.filtered_proteins:
                    continue
                
                if protein not in protein_pathway_count:
                    protein_pathway_count[protein] = 0
                protein_pathway_count[protein] += 1
        
        # 识别枢纽蛋白
        hub_proteins = {
            protein for protein, count in protein_pathway_count.items()
            if count >= self.hub_threshold
        }
        
        print(f"  Total proteins: {len(protein_pathway_count)}")
        print(f"  Hub proteins: {len(hub_proteins)}")
        if len(hub_proteins) > 0:
            hub_counts = [protein_pathway_count[p] for p in hub_proteins]
            print(f"  Hub protein pathway counts: min={min(hub_counts)}, "
                  f"max={max(hub_counts)}, avg={sum(hub_counts)/len(hub_counts):.1f}")
        
        return hub_proteins
    
    def get_dynamic_threshold(self, layer_level, n_levels):
        """
        根据层级获取动态阈值
        
        策略: 高层通路更抽象，需要更多共享蛋白质
              低层通路更具体，少量共享就有意义
        
        Args:
            layer_level: 当前层级（从0开始）
            n_levels: 总层数
        
        Returns:
            int: 该层级的最小共享蛋白数阈值
        """
        if not self.use_dynamic_threshold:
            return self.min_shared_proteins
        
        # 高层（接近root）需要更多共享蛋白
        if layer_level <= n_levels // 2:
            return max(5, self.min_shared_proteins)
        else:
            return self.min_shared_proteins
    
    def select_shared_proteins(self, intersect, layer_level, n_levels):
        """
        根据策略选择共享蛋白质
        
        策略:
        1. 优先使用非枢纽蛋白
        2. 如果非枢纽蛋白不足，添加部分枢纽蛋白
        
        Args:
            intersect: 共享蛋白质集合
            layer_level: 当前层级
            n_levels: 总层数
        
        Returns:
            set: 选择的蛋白质集合
        """
        def rank(protein):
            pathway_count = sum(
                1 for pathway_id, proteins in self.pathway_protein_dict.items()
                if pathway_id in self.filtered_pathways and protein in proteins
            )
            return pathway_count, protein

        if not self.filter_hub_proteins or len(self.hub_proteins) == 0:
            return set(sorted(intersect, key=rank)[:self.max_mediators_per_edge])
        
        # 获取动态阈值
        threshold = self.get_dynamic_threshold(layer_level, n_levels)
        
        # 分离枢纽蛋白和非枢纽蛋白
        non_hub_shared = intersect - self.hub_proteins
        hub_shared = intersect & self.hub_proteins
        
        # 优先使用非枢纽蛋白
        if len(non_hub_shared) >= threshold:
            selected_proteins = non_hub_shared
        else:
            # 非枢纽蛋白不足，添加部分枢纽蛋白
            selected_proteins = non_hub_shared.copy()
            
            # 按参与通路数排序，选择参与通路较少的枢纽蛋白
            if len(hub_shared) > 0:
                # 计算每个枢纽蛋白参与的通路数
                hub_counts = {}
                for protein in hub_shared:
                    count = sum(
                        1 for pathway_id, proteins in self.pathway_protein_dict.items()
                        if pathway_id in self.filtered_pathways and protein in proteins
                    )
                    hub_counts[protein] = count
                
                # 按参与通路数排序，选择前k个
                sorted_hubs = sorted(hub_counts.items(), key=lambda x: x[1])
                k = min(self.max_hub_proteins, len(sorted_hubs))
                selected_hubs = {protein for protein, _ in sorted_hubs[:k]}
                selected_proteins.update(selected_hubs)
            
        return set(
            sorted(selected_proteins, key=rank)[:self.max_mediators_per_edge]
        )
    
    def get_reactome_networkx(self):
        """
        构建筛选后的Reactome通路网络
        只包含与输入基因相关的通路
        """
        print("\nBuilding filtered Reactome network...")
        
        hierarchy = self.reactome.hierarchy
        # 过滤人类通路
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        
        # 只保留筛选后的通路
        filtered_hierarchy = human_hierarchy[
            human_hierarchy['child'].isin(self.filtered_pathways) &
            human_hierarchy['parent'].isin(self.filtered_pathways)
        ]
        
        print(f"  Original edges: {len(human_hierarchy)}")
        print(f"  Filtered edges: {len(filtered_hierarchy)}")
        
        net = nx.from_pandas_edgelist(filtered_hierarchy, 'child', 'parent', 
                                     create_using=nx.DiGraph())
        net.name = 'reactome_filtered'
        
        # 添加root节点
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)
        
        print(f"  Network nodes: {net.number_of_nodes()}")
        print(f"  Network edges: {net.number_of_edges()}")
        print(f"  Root connections: {len(roots)}")
        
        return net
    
    def insert_protein_between_pathways(self, net, n_levels):
        """
        插入蛋白质节点，使用筛选后的蛋白质集合和动态阈值策略
        
        改进:
        1. 只使用筛选后的蛋白质
        2. 如果使用KG，优先使用KG中的蛋白质关系
        3. 使用分层动态阈值策略
        4. 过滤枢纽蛋白，优先使用非枢纽蛋白
        
        Args:
            net: 网络图
            n_levels: 总层数（用于动态阈值计算）
        """
        print("\n=== Inserting Filtered Proteins Between Pathways ===")
        print(f"Threshold Strategy:")
        print(f"  Base min_shared_proteins: {self.min_shared_proteins}")
        print(f"  Base min_shared_ratio: {self.min_shared_ratio}")
        print(f"  Use dynamic threshold: {self.use_dynamic_threshold}")
        print(f"  Filter hub proteins: {self.filter_hub_proteins}")
        if self.filter_hub_proteins:
            print(f"  Hub threshold: {self.hub_threshold}")
        print(f"  Max hub proteins per edge: {self.max_hub_proteins}")
        print(f"  Max mediators per edge: {self.max_mediators_per_edge}")
        
        add_edges_list = []
        remove_edges_list = []
        
        # 统计信息
        total_edges = 0
        edges_with_proteins = 0
        total_proteins_inserted = 0
        proteins_from_kg = 0
        proteins_from_ppi = 0
        hub_proteins_used = 0
        edges_filtered_by_threshold = 0
        edges_filtered_by_ratio = 0
        
        # 按层级统计
        layer_stats = {}
        
        for edge in list(net.edges):
            start_pathway_s = edge[0]
            end_pathway_s = edge[1]
            
            if start_pathway_s == 'root':
                continue
            
            total_edges += 1
            
            # 计算终点通路的层级深度
            end_pos = len(nx.shortest_path(net, source='root', target=end_pathway_s))
            layer_level = end_pos - 1  # 层级从0开始
            
            # 初始化层级统计
            if layer_level not in layer_stats:
                layer_stats[layer_level] = {
                    'total': 0, 'with_proteins': 0, 'proteins': 0,
                    'filtered_threshold': 0, 'filtered_ratio': 0
                }
            layer_stats[layer_level]['total'] += 1
            
            # 移除_copy后缀
            start_pathway = re.sub('_copy.*', '', start_pathway_s)
            end_pathway = re.sub('_copy.*', '', end_pathway_s)
            
            # 获取两个通路的蛋白质集合
            start_proteins = set(self.pathway_protein_dict.get(start_pathway, []))
            end_proteins = set(self.pathway_protein_dict.get(end_pathway, []))
            
            # 只保留筛选后的蛋白质
            start_proteins = start_proteins.intersection(self.filtered_proteins)
            end_proteins = end_proteins.intersection(self.filtered_proteins)
            
            # 计算交集
            shared_proteins = start_proteins.intersection(end_proteins)
            ppi_bridge_proteins = self.find_ppi_bridge_proteins(
                start_proteins, end_proteins
            )
            candidates = shared_proteins | ppi_bridge_proteins
            proteins_from_ppi += len(ppi_bridge_proteins - shared_proteins)

            dynamic_threshold = self.get_dynamic_threshold(layer_level, n_levels)
            
            # KG completion is supplementary and is activated only when the
            # direct shared/PPI-supported evidence does not meet the
            # level-specific minimum.
            if (
                self.use_kg
                and self.kg_protein_protein is not None
                and len(candidates) < dynamic_threshold
            ):
                # 查找KG中连接这两个通路蛋白质的桥接蛋白
                kg_bridge_proteins = self.find_kg_bridge_proteins(
                    start_proteins, end_proteins
                )
                if len(kg_bridge_proteins) > 0:
                    proteins_from_kg += len(kg_bridge_proteins - candidates)
                    candidates |= kg_bridge_proteins
            
            # 应用动态阈值策略
            if len(candidates) > 0:
                # 策略1: 基础阈值 - 最小共享蛋白数
                if len(candidates) < dynamic_threshold:
                    edges_filtered_by_threshold += 1
                    layer_stats[layer_level]['filtered_threshold'] += 1
                    continue
                
                # 策略2: 基础阈值 - 最小共享比例
                min_pathway_size = min(len(start_proteins), len(end_proteins))
                if min_pathway_size > 0:
                    shared_ratio = len(candidates) / min_pathway_size
                    if shared_ratio < self.min_shared_ratio:
                        edges_filtered_by_ratio += 1
                        layer_stats[layer_level]['filtered_ratio'] += 1
                        continue
                
                # 策略3: 枢纽蛋白处理
                selected_proteins = self.select_shared_proteins(
                    candidates, layer_level, n_levels
                )

                if len(selected_proteins) < dynamic_threshold:
                    edges_filtered_by_threshold += 1
                    layer_stats[layer_level]['filtered_threshold'] += 1
                    continue
                
                # 统计枢纽蛋白使用情况
                if self.filter_hub_proteins:
                    hub_count = len(selected_proteins & self.hub_proteins)
                    if hub_count > 0:
                        hub_proteins_used += hub_count
                
                edges_with_proteins += 1
                total_proteins_inserted += len(selected_proteins)
                layer_stats[layer_level]['with_proteins'] += 1
                layer_stats[layer_level]['proteins'] += len(selected_proteins)
                
                # 插入蛋白质节点
                for protein in selected_proteins:
                    protein_node = protein + '-' + str(end_pos - 0.5)
                    add_edges_list.append((start_pathway_s, protein_node))
                    add_edges_list.append((protein_node, end_pathway_s))
                
                remove_edges_list.append((start_pathway_s, end_pathway_s))
        
        # 批量更新网络
        net.add_edges_from(add_edges_list)
        net.remove_edges_from(remove_edges_list)
        
        # 打印统计信息
        print(f"\nOverall Statistics:")
        print(f"  Total pathway-pathway edges: {total_edges}")
        retained_pct = (edges_with_proteins / total_edges * 100) if total_edges else 0.0
        print(f"  Edges with retained mediators: {edges_with_proteins} ({retained_pct:.1f}%)")
        print(f"  Edges filtered by threshold: {edges_filtered_by_threshold}")
        print(f"  Edges filtered by ratio: {edges_filtered_by_ratio}")
        print(f"  Total proteins inserted: {total_proteins_inserted}")
        if edges_with_proteins > 0:
            print(f"  Average proteins per edge: {total_proteins_inserted/edges_with_proteins:.1f}")
        if self.use_kg:
            print(f"  Proteins from KG bridges: {proteins_from_kg}")
        print(f"  Additional proteins supported by cross-set PPI: {proteins_from_ppi}")
        if self.filter_hub_proteins:
            print(f"  Hub proteins used: {hub_proteins_used}")
        print(f"  New edges added: {len(add_edges_list)}")
        print(f"  Old edges removed: {len(remove_edges_list)}")
        
        # 打印层级统计
        print(f"\nLayer-wise Statistics:")
        for layer in sorted(layer_stats.keys()):
            stats = layer_stats[layer]
            threshold = self.get_dynamic_threshold(layer, n_levels)
            print(f"  Layer {layer} (threshold={threshold}):")
            print(f"    Total edges: {stats['total']}")
            layer_pct = (
                stats['with_proteins'] / stats['total'] * 100
                if stats['total'] else 0.0
            )
            print(f"    With proteins: {stats['with_proteins']} ({layer_pct:.1f}%)")
            if stats['with_proteins'] > 0:
                print(f"    Avg proteins: {stats['proteins']/stats['with_proteins']:.1f}")
            print(f"    Filtered by threshold: {stats['filtered_threshold']}")
            print(f"    Filtered by ratio: {stats['filtered_ratio']}")
        
        return net

    def find_ppi_bridge_proteins(self, start_proteins, end_proteins):
        """Return pathway proteins participating in cross-set STRING edges."""
        if self.ppi_graph is None:
            return set()

        bridge_proteins = set()
        end_set = set(end_proteins)
        for protein in start_proteins:
            if protein not in self.ppi_graph:
                continue
            cross_neighbors = set(self.ppi_graph.neighbors(protein)) & end_set
            if cross_neighbors:
                bridge_proteins.add(protein)
                bridge_proteins.update(cross_neighbors)
        return bridge_proteins.intersection(self.filtered_proteins)
    
    def find_kg_bridge_proteins(self, start_proteins, end_proteins):
        """
        从KG中查找连接两组蛋白质的桥接蛋白
        
        策略: 查找在KG中同时连接start_proteins和end_proteins的蛋白质
        """
        if self.kg_protein_protein is None:
            return set()
        
        kg_graph = nx.from_pandas_edgelist(
            self.kg_protein_protein, 'child', 'parent', create_using=nx.Graph()
        )

        def two_hop_context(seeds):
            context = set(seeds)
            for seed in seeds:
                if seed in kg_graph:
                    context.update(
                        nx.single_source_shortest_path_length(
                            kg_graph, seed, cutoff=2
                        ).keys()
                    )
            return context

        bridge_proteins = (
            two_hop_context(start_proteins)
            & two_hop_context(end_proteins)
        )
        
        # 只保留在筛选后的蛋白质集合中的
        bridge_proteins = bridge_proteins.intersection(self.filtered_proteins)
        
        return bridge_proteins

    
    def get_completed_network(self, n_levels):
        """获取补全后的网络"""
        G = complete_pathway_network(self.netx, n_leveles=n_levels)
        return G
    
    def get_layers(self, n_levels, direction='root_to_leaf'):
        """
        获取知识增强的混合网络层级结构
        
        Args:
            n_levels: 原始Reactome层数
            direction: 'root_to_leaf' 或 'leaf_to_root'
        
        Returns:
            layers: 层级列表
        """
        print(f"\n{'='*70}")
        print(f"Building KG-Enhanced Hybrid Network (n_levels={n_levels})")
        print(f"{'='*70}\n")
        
        if direction == 'root_to_leaf':
            # Step 1: 获取补全后的网络
            net = self.get_completed_network(n_levels)
            print(f"Completed network: {net.number_of_nodes()} nodes, {net.number_of_edges()} edges")
            
            # Step 2: 插入筛选后的蛋白质（使用动态阈值策略）
            net = self.insert_protein_between_pathways(net, n_levels)
            print(f"Hybrid network: {net.number_of_nodes()} nodes, {net.number_of_edges()} edges")
            
            # Step 3: 提取层级
            layers = get_layers_from_net(net, n_levels + n_levels - 1)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]
        
        # Step 4: 添加基因层（只包含输入的基因）
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]
        
        gene_layer = {}
        missing_pathways = []
        
        if self.use_kg and self.kg_gene_pathway is not None:
            # 使用KG增强的基因-通路映射
            print("\nBuilding gene layer using KG mapping...")
            for pathway in terminal_nodes:
                pathway_name = re.sub('_copy.*', '', pathway)
                # 从KG中获取该通路的基因
                genes = self.kg_gene_pathway[
                    self.kg_gene_pathway['pathway'] == pathway_name
                ]['gene'].unique()
                
                # 只保留在输入基因列表中的
                genes = [g for g in genes if g in self.genes]
                
                if len(genes) == 0:
                    # 如果KG中没有，回退到标准映射
                    genes_df = self.reactome.pathway_genes
                    genes = genes_df[genes_df['group'] == pathway_name]['gene'].unique()
                    genes = [g for g in genes if g in self.genes]
                    
                    if len(genes) == 0:
                        missing_pathways.append(pathway_name)
                
                gene_layer[pathway_name] = genes
        else:
            # 使用标准映射
            print("\nBuilding gene layer using standard mapping...")
            genes_df = self.reactome.pathway_genes
            for pathway in terminal_nodes:
                pathway_name = re.sub('_copy.*', '', pathway)
                genes = genes_df[genes_df['group'] == pathway_name]['gene'].unique()
                # 只保留在输入基因列表中的
                genes = [g for g in genes if g in self.genes]
                
                if len(genes) == 0:
                    missing_pathways.append(pathway_name)
                
                gene_layer[pathway_name] = genes
        
        layers.append(gene_layer)
        
        # 打印最终统计
        print(f"\n{'='*70}")
        print("Final KG-Enhanced Network Structure:")
        print(f"{'='*70}")
        print(f"Total layers: {len(layers)}")
        for i, layer in enumerate(layers):
            node_count = len(layer)
            if i < len(layers) - 1:
                # 不是基因层
                edge_count = sum(len(v) for v in layer.values())
                print(f"  Layer {i}: {node_count} nodes, {edge_count} edges")
            else:
                # 基因层
                gene_count = sum(len(v) for v in layer.values())
                print(f"  Layer {i} (genes): {node_count} pathways, {gene_count} genes")
        
        if len(missing_pathways) > 0:
            print(f"\nWarning: {len(missing_pathways)} pathways have no genes")
        
        # 计算稀疏度
        self.print_sparsity_stats(layers)
        
        return layers
    
    def print_sparsity_stats(self, layers):
        """打印网络稀疏度统计"""
        print(f"\n{'='*70}")
        print("Network Sparsity Statistics:")
        print(f"{'='*70}")
        
        total_possible_edges = 0
        total_actual_edges = 0
        
        for i, layer in enumerate(layers[:-1]):  # 不包括基因层
            n_source = len(layer)
            if i + 1 < len(layers):
                n_target = len(layers[i + 1])
                possible_edges = n_source * n_target
                actual_edges = sum(len(v) for v in layer.values())
                
                sparsity = (1 - actual_edges / possible_edges) * 100 if possible_edges > 0 else 0
                
                print(f"  Layer {i} → {i+1}:")
                print(f"    Possible edges: {possible_edges}")
                print(f"    Actual edges: {actual_edges}")
                print(f"    Sparsity: {sparsity:.2f}%")
                
                total_possible_edges += possible_edges
                total_actual_edges += actual_edges
        
        if total_possible_edges > 0:
            overall_sparsity = (1 - total_actual_edges / total_possible_edges) * 100
            print(f"\n  Overall sparsity: {overall_sparsity:.2f}%")
            print(f"  Compression ratio: {total_possible_edges / total_actual_edges:.2f}x")
