import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.jumping_knowledge import JumpingKnowledge
from operations import APPNP, make_degree, ConvBlock, make_multihop_edges, BondEncoder, OGBMolEmbedding, GlobalPool


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_tasks):
        super(Net, self).__init__()
        self.config = config
        self.model = Model(config, num_tasks)
        print (self.model) 

    def forward(self, batched_data, perturb=None):
        return self.model(batched_data, perturb)

    def __repr__(self):
        return self.__class__.__name__


class Model(nn.Module):
    def __init__(self, config, num_tasks):
        super(Model, self).__init__()
        self.config = config
        virtual_node = True if config.virtual_node == 'true' else False
        self.k = config.k
        self.conv_type = config.methods
        hidden = config.hidden
        layers = config.layers
        out_dim = num_tasks
        self.degree = True if config.degree == 'true' else False
        convs = [ConvBlock(hidden,
                           dropout=config.dropout,
                           virtual_node=virtual_node,
                           k=min(i + 1, self.k),
                           conv_type=self.conv_type,
                           edge_embedding=BondEncoder(emb_dim=hidden))
                 for i in range(layers - 1)]
        convs.append(ConvBlock(hidden,
                               dropout=config.dropout,
                               virtual_node=virtual_node,
                               virtual_node_agg=False,  # on last layer, use but do not update virtual node
                               last_layer=True,
                               k=min(layers, self.k),
                               conv_type=self.conv_type,
                               edge_embedding=BondEncoder(emb_dim=hidden)))
        self.main = nn.Sequential(
            OGBMolEmbedding(hidden, embed_edge=False, 
                x_as_list=(self.conv_type=='gin+' or self.conv_type=='gin++'), degree=self.degree),
            *convs)
        if config.appnp == 'true':
            self.aggregate = nn.Sequential(
                APPNP(K=5, alpha=0.8),
                GlobalPool(config.pooling, hidden=hidden),
                nn.Linear(hidden, out_dim)
            )
        else:
            self.aggregate = nn.Sequential(
                GlobalPool(config.pooling, hidden=hidden),
                nn.Linear(hidden, out_dim)
            )
        self.virtual_node = virtual_node
        if self.virtual_node:
            self.v0 = nn.Parameter(torch.zeros(1, hidden), requires_grad=True)

    def forward(self, data, perturb=None):
        data = make_degree(data)
        data = make_multihop_edges(data, self.k)
        if self.virtual_node:
            data.virtual_node = self.v0.expand(data.num_graphs, self.v0.shape[-1])
        data.perturb = perturb
        g = self.main(data)
        if self.conv_type == 'gin+' or self.conv_type == 'gin++':
            g.x = g.x[0]
        if self.training:
            return self.aggregate(g)/self.config.T
        else:    
            return self.aggregate(g)
