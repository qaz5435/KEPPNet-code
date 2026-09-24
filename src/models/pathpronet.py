from logging import getLogger
from copy import deepcopy
from src.models.custom.builder_utils import get_layer_maps
from src.models.custom.layer_custom import CustomizedLinear, Diagonal, GraphConvolutionLayer
from src.config.configuration import Config
from src.data.dataset import BnetData
from src.data.processor.protein.ppi import Protein


import torch
import numpy as np
import pandas as pd
from torch import nn, tanh, sigmoid


class Model(nn.Module):

    def __init__(self, config:Config, dataset:BnetData):
        super(Model, self).__init__()

        # self.model_params = param['model_params']
        # self.data_params = self.model_params['data_params']
        self.config = config  # 保存config对象以便后续使用
        self.use_bias = config['use_bias']
        self.trainable_mask = config['trainable_mask']
        self.full_train = config['full_train']
        # 保存原始的 n_hidden_layers（Reactome 层数）
        self.reactome_levels = config['n_hidden_layers']
        # 暂时设置为原始值，后面会根据实际 maps 数量更新
        self.n_hidden_layers = config['n_hidden_layers']
        self.add_unk_genes = config['add_unk_genes']
        self.dropout = config['dropout']
        self.network_flag = config['network']
        self.gcn = config['gcn']
        self.dense = config['dense'] if config['dense'] else False

        self.logger = getLogger()
        self.get_model_data(dataset)
        self.logger.debug(f"input dim is {self.in_size}")
        
        
        self.dropout1 = nn.Dropout(p=self.dropout[0])
        self.dropout2 = nn.Dropout(p=self.dropout[1])

        # each node in the next layer is connected to exactly three nodes of the input
        # layer representing mutations, copy number amplification and copy number deletions.
        self.h0 = Diagonal(self.in_size, self.maps[0].shape[0], self.use_bias, self.trainable_mask)  # sparse layer 27687*9229
        if self.gcn == True:
            self.g0 = GraphConvolutionLayer(self.maps[0].shape[0], self.maps[0].shape[0])
            self.g1 = GraphConvolutionLayer(self.maps[0].shape[0], self.maps[0].shape[0])

        self.hidden_layers = nn.ModuleList([
            CustomizedLinear(
                input_features=self.maps[i].shape[0], 
                output_features=self.maps[i].shape[1], 
                bias=self.use_bias, 
                mask=np.array(self.maps[i].values) if not self.dense else None, 
                trainable_mask=self.trainable_mask,
                full_train=self.full_train,
            ) for i in range(self.n_hidden_layers)
        ])
        # # sparse layer 9227*1387
        # self.h1 = CustomizedLinear(self.maps[0].shape[0], self.maps[0].shape[1], self.use_bias,
        #                             np.array(self.maps[0].values), self.trainable_mask)
        # # sparse layer 1387*1066
        # self.h2 = CustomizedLinear(self.maps[1].shape[0], self.maps[1].shape[1], self.use_bias,
        #                             np.array(self.maps[1].values), self.trainable_mask)
        # # sparse layer 1066*447
        # self.h3 = CustomizedLinear(self.maps[2].shape[0], self.maps[2].shape[1], self.use_bias,
        #                             np.array(self.maps[2].values), self.trainable_mask)
        # # sparse layer 447*147
        # self.h4 = CustomizedLinear(self.maps[3].shape[0], self.maps[3].shape[1], self.use_bias,
        #                             np.array(self.maps[3].values), self.trainable_mask)
        # # sparse layer 147*26
        # self.h5 = CustomizedLinear(self.maps[4].shape[0], self.maps[4].shape[1], self.use_bias,
        #                             np.array(self.maps[4].values), self.trainable_mask)
        self.linears = nn.ModuleList([nn.Linear(self.maps[0].shape[0], 1)])
        self.linears.extend([
            nn.Linear(self.maps[i].shape[1], 1)
            for i in range(self.n_hidden_layers)
        ])
        # self.l1 = nn.Linear(self.maps[0].shape[0], 1)
        # self.l2 = nn.Linear(self.maps[0].shape[1], 1)
        # self.l3 = nn.Linear(self.maps[1].shape[1], 1)
        # self.l4 = nn.Linear(self.maps[2].shape[1], 1)
        # self.l5 = nn.Linear(self.maps[3].shape[1], 1)
        # self.l6 = nn.Linear(self.maps[4].shape[1], 1)

    def forward(self, input):
        out = tanh(self.h0(input))

        if self.gcn == True:
            out = self.g0(out, self.gene_adj_matrix)
            out = self.dropout1(out)
            out = self.g1(out, self.gene_adj_matrix)
            out = self.dropout1(out)

        o1 = sigmoid(self.linears[0](out))
        out = self.dropout1(out)


        outputs_per_layer = [o1]
        for i in range(self.n_hidden_layers):
            out = tanh(self.hidden_layers[i](out))
            outputs_per_layer.append(
                sigmoid(self.linears[i + 1](out))
            )
            out = self.dropout2(out)
            
        return torch.concat(outputs_per_layer, dim=1)

    def get_model_data(self, dataset):
        """get mask matrix that sparseNn have."""
        x, y, info, cols = dataset.get_data()
        self.in_size = cols.shape[0]
        self.logger.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

        if hasattr(cols, 'levels'):
            genes = cols.levels[0]
        else:
            genes = cols
        self.feature_names = {}
        self.feature_names['inputs'] = cols
        if self.n_hidden_layers > 0:
            # 传递 Reactome 层数给 get_layer_maps
            # get_layers 会返回混合网络的所有层（包括基因层）
            maps = get_layer_maps(genes, self.reactome_levels, 'root_to_leaf', self.add_unk_genes, self.network_flag, self.config)

            self.maps = maps
            # 更新 n_hidden_layers 为实际返回的层数 - 1
            # 因为 maps[0] 被 h0 使用，剩下的被 hidden_layers 使用
            self.n_hidden_layers = len(maps) - 1
            self.logger.info(f'Network has {len(maps)} layers (including gene layer), using {self.n_hidden_layers} hidden layers')
        for i, _maps in enumerate(maps):
            self.feature_names[f'h{i}'] = _maps.index
        
        # get adjacency matrix of gene
        # self.gene_adj_matrix = get_gene_adjacency_matrix() 
        assert list(genes) == list(self.maps[0].index), 'some genes don\'t match!'
        if self.gcn == True:
            protein = Protein(genes)
            id_hier = protein.hierarchy.copy()
            genes_id = [protein.name2id[name] for name in genes if name in protein.name2id.keys()]
            id_hier = id_hier[
                (id_hier['child'].isin(genes_id)) &
                (id_hier['parent'].isin(genes_id))
            ]
            name_hier = id_hier.copy()
            name_hier['child'] = id_hier['child'].replace(protein.id2name)
            name_hier['parent'] = id_hier['parent'].replace(protein.id2name)
            df = pd.pivot_table(
                data=name_hier, 
                values='combined_score',
                index='child',
                columns='parent',
                aggfunc='count'
            )

            df = df.reindex(genes)
            df = df.reindex(columns=genes)
            df_symmetric = df.fillna(df.T)  # symmetric matrix
            np.fill_diagonal(df_symmetric.values, 1)  # diagonal
            self.gene_adj_matrix = torch.Tensor(df_symmetric.fillna(0).values)
        else:
            self.gene_adj_matrix = None
