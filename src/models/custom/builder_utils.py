import itertools

import numpy as np
import pandas as pd

from src.data.processor.pathways.reactome import ReactomeNetwork
from src.data.processor.protein.ppi import ProteinNetwork
from src.data.processor.pathway_protein.pathway_protein import ReactomeProteinNetwork
from src.data.processor.pathway_protein.pathway_protein_kg_enhanced import ReactomeProteinNetworkKGEnhanced


def _dataset_to_cancer_type(config):
    """Map experiment-dataset identifiers to the data-directory names.

    The public experiment configurations use ``prostate_kg`` and ``breast_kg``
    to distinguish the KG-enhanced model variants.  The derived KG resources,
    however, are stored under the cancer names ``prostate`` and ``breast``.
    Keeping this translation in one place prevents configuration names from
    leaking into file paths such as ``data/protein/prostate_id2id_2hop.csv``.
    """
    if config is None or config['dataset'] is None:
        return None

    dataset_name = str(config['dataset'])
    return {
        'prostate': 'prostate',
        'prostate_kg': 'prostate',
        'breast': 'breast',
        'breast_kg': 'breast',
    }.get(dataset_name, dataset_name)


def get_map_from_layer(layer_dict):
    pathways = list(layer_dict.keys()) #word of python3
    print('pathways', len(pathways))
    genes = list(itertools.chain.from_iterable(layer_dict.values()))
    genes = list(np.unique(genes))
    print('genes', len(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1 #It's like an adjacency matrix, row is pathway_index, col is genes_index

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    # for k, v in layer_dict.items():
    #     print k, v
    #     df.loc[k,v] = 1
    # df= df.fillna(0)
    return df.T


def get_layer_maps(genes, n_levels, direction, add_unk_genes, network_flag, config=None):
    """
    获取网络层级映射
    
    Args:
        genes: 基因列表
        n_levels: 层数
        direction: 方向 ('root_to_leaf' 或 'leaf_to_root')
        add_unk_genes: 是否添加未知基因
        network_flag: 网络类型标志
        config: 配置对象（用于获取额外参数）
    """
    if network_flag == 'reactome':
        reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    elif network_flag == 'protein':
        # This ablation uses direct STRING evidence only.  Passing a cancer
        # name here would silently intersect the network with an optional KG
        # table and would no longer be a true non-KG baseline.
        reactome_layers = ProteinNetwork(genes, cancer_type=None).get_layers(n_levels, direction)
    elif network_flag == 'reactome_protein':
        reactome_layers = ReactomeProteinNetwork(genes, cancer_type=None).get_layers(n_levels, direction)
    elif network_flag == 'reactome_protein_kg':
        # KG增强版本 - 从config中获取参数
        if config is not None:
            # 从配置中获取癌症类型
            dataset_name = config['dataset'] if config['dataset'] is not None else 'breast_kg'
            cancer_type = _dataset_to_cancer_type(config) or 'breast'
            
            # 获取KG和阈值策略参数
            use_kg = config['use_kg'] if config['use_kg'] is not None else True
            min_shared_proteins = config['min_shared_proteins'] if config['min_shared_proteins'] is not None else 2
            min_shared_ratio = config['min_shared_ratio'] if config['min_shared_ratio'] is not None else 0.05
            use_dynamic_threshold = config['use_dynamic_threshold'] if config['use_dynamic_threshold'] is not None else True
            filter_hub_proteins = config['filter_hub_proteins'] if config['filter_hub_proteins'] is not None else True
            hub_threshold = config['hub_threshold'] if config['hub_threshold'] is not None else 50
            max_hub_proteins = config['max_hub_proteins'] if config['max_hub_proteins'] is not None else 3
            max_mediators_per_edge = config['max_mediators_per_edge'] if config['max_mediators_per_edge'] is not None else 10
            
            print(f"\n{'='*70}")
            print(f"Using KG-Enhanced Hybrid Network")
            print(f"  Dataset config: {dataset_name}")
            print(f"  Cancer resource directory: {cancer_type}")
            print(f"  Use KG: {use_kg}")
            print(f"  Min shared proteins: {min_shared_proteins}")
            print(f"  Min shared ratio: {min_shared_ratio}")
            print(f"  Dynamic threshold: {use_dynamic_threshold}")
            print(f"  Filter hub proteins: {filter_hub_proteins}")
            if filter_hub_proteins:
                print(f"  Hub threshold: {hub_threshold}")
                print(f"  Max hub proteins: {max_hub_proteins}")
                print(f"  Max mediators per edge: {max_mediators_per_edge}")
            print(f"{'='*70}\n")
            
            reactome_layers = ReactomeProteinNetworkKGEnhanced(
                genes=genes,
                cancer_type=cancer_type,
                use_kg=use_kg,
                use_processed_data=True,
                min_shared_proteins=min_shared_proteins,
                min_shared_ratio=min_shared_ratio,
                use_dynamic_threshold=use_dynamic_threshold,
                filter_hub_proteins=filter_hub_proteins,
                hub_threshold=hub_threshold,
                max_hub_proteins=max_hub_proteins,
                max_mediators_per_edge=max_mediators_per_edge
            ).get_layers(n_levels, direction)
        else:
            # 使用默认参数
            print("Warning: No config provided, using default parameters for KG-enhanced network")
            reactome_layers = ReactomeProteinNetworkKGEnhanced(
                genes=genes,
                cancer_type='breast',
                use_kg=True
            ).get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print('layer #', i)
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        print('filtered_map', filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print('filtered_map', filter_df.shape)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

        # UNK, add a node for genes without known reactome annotation
        if add_unk_genes:
            print('UNK ')
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, 'UNK'] = 1
        ####

        filtered_map = filtered_map.fillna(0)
        print('filtered_map', filter_df.shape)
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        print('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps


def shuffle_genes_map(mapp):
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    # print('shuffling the map')
    # mapp = mapp.T
    # np.random.shuffle(mapp)
    # mapp= mapp.T
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    print('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    print('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    print('random map ones_ratio {}'.format(ones_ratio))
    return mapp
