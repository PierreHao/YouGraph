import torch
import torch.nn.functional as F
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from conv import GinConv, ExpC, CombC, ExpC_star, CombC_star
from torch.nn import PReLU, ReLU, BatchNorm1d as BN


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_class):
        super(Net, self).__init__()
        self.node_encoder = torch.nn.Embedding(1, config.hidden)

        self.convs = torch.nn.ModuleList()
        self.config = config
        if config.methods == 'GIN':
            for i in range(config.layers):
                self.convs.append(GinConv(config.hidden, config.variants))
        elif config.methods[:2] == 'EB':
            for i in range(config.layers):
                self.convs.append(ExpC(config.hidden,
                                                 int(config.methods[2:]),
                                                 config.variants))
        elif config.methods[:2] == 'EA':
            for i in range(config.layers):
                self.convs.append(ExpC_star(config.hidden,
                                                 int(config.methods[2:]),
                                                 config.variants))
        elif config.methods == 'CB':
            for i in range(config.layers):
                self.convs.append(CombC(config.hidden, config.variants))
        elif config.methods == 'CA':
            for i in range(config.layers):
                self.convs.append(CombC_star(config.hidden, config.variants))
        else:
            raise ValueError('Undefined gnn layer called {}'.format(config.methods))

        self.JK = JumpingKnowledge(config.JK)

        if config.JK == 'cat':
            self.graph_pred_linear = torch.nn.Linear(config.hidden * config.layers, num_class)
        elif config.JK == 'multi-softmax':
            self.graph_pred_linears = torch.nn.ModuleList()
            for i in range(config.layers):
                self.graph_pred_linears.append(torch.nn.Linear(config.hidden, num_class))
        else:
            self.graph_pred_linear = torch.nn.Linear(config.hidden, num_class)

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool
        elif config.pooling == 'max':
            self.pool = global_max_pool
        # there is a bug, not work
        elif config.pooling == "attention":
            if config.JK == 'cat':
                emb_dim = config.hidden * config.layers
            else:
                emb_dim = config.hidden
            self.pool = GlobalAttention(gate_nn = 
                    torch.nn.Sequential(
                        torch.nn.Linear(emb_dim, emb_dim * 2)
                        , torch.nn.BatchNorm1d(2*emb_dim)
                        , torch.nn.ReLU()
                        , torch.nn.Linear(2*emb_dim, 1)))

        self.dropout = config.dropout

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x = self.node_encoder(x) + perturb if perturb is not None else self.node_encoder(x)
        if self.config.input_dropout == 'true':
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs = []
        for i,conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            xs += [x]

        nr = self.JK(xs)
        nr = F.dropout(nr, p=self.dropout, training=self.training)
        h_graph = self.pool(nr, batched_data.batch)
        if self.config.JK == 'multi-softmax':
            return [self.graph_pred_linears[i](h_graph[:,self.config.hidden*i:self.config.hidden*(i+1)]) for i in range(self.config.layers)]
        else:
            return self.graph_pred_linear(h_graph) / self.config.T

    def __repr__(self):
        return self.__class__.__name__



class VirtualnodeNet(torch.nn.Module):
    def __init__(self,
                 config,
                 num_class):
        super(VirtualnodeNet, self).__init__()
        self.node_encoder = torch.nn.Embedding(1, config.hidden)

        self.convs = torch.nn.ModuleList()
        self.config = config
        if config.methods == 'GIN':
            for i in range(config.layers):
                self.convs.append(GinConv(config.hidden, config.variants))
        elif config.methods[:2] == 'EB':
            for i in range(config.layers):
                self.convs.append(ExpC(config.hidden,
                                                 int(config.methods[2:]),
                                                 config.variants))
        elif config.methods[:2] == 'EA':
            for i in range(config.layers):
                self.convs.append(ExpC_star(config.hidden,
                                                 int(config.methods[2:]),
                                                 config.variants))
        elif config.methods == 'CB':
            for i in range(config.layers):
                self.convs.append(CombC(config.hidden, config.variants))
        elif config.methods == 'CA':
            for i in range(config.layers):
                self.convs.append(CombC_star(config.hidden, config.variants))
        else:
            raise ValueError('Undefined gnn layer called {}'.format(config.methods))

        self.JK = JumpingKnowledge(config.JK)

        if config.JK == 'cat':
            self.graph_pred_linear = torch.nn.Linear(config.hidden * config.layers, num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(config.hidden, num_class)

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool
        elif config.pooling == 'max':
            self.pool = global_max_pool

        self.dropout = config.dropout
        # virtualnode
        self.virtualnode_embedding = torch.nn.Embedding(1, config.hidden)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for layer in range(config.layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(
                torch.nn.Linear(config.hidden, 2*config.hidden)
                ,torch.nn.BatchNorm1d(2*config.hidden)
                ,torch.nn.PReLU()
                ,torch.nn.Linear(2*config.hidden, config.hidden)
                ,torch.nn.BatchNorm1d(config.hidden)
                ,torch.nn.PReLU()
                ))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        tmp = self.node_encoder(x) + perturb if perturb is not None else self.node_encoder(x)
        xs = [tmp]
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        for i, conv in enumerate(self.convs):
            xs[i] = xs[i] + virtualnode_embedding[batch]
            x = conv(xs[i], edge_index, edge_attr)
            xs.append(x)
        
            if i < self.config.layers-1:
                virtualnode_embedding_tmp = global_add_pool(xs[i], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_tmp), p=self.dropout, training=self.training)

        nr = self.JK(xs[1:])
        nr = F.dropout(nr, p=self.dropout, training=self.training)
        h_graph = self.pool(nr, batched_data.batch)
        return self.graph_pred_linear(h_graph) / self.config.T

    def __repr__(self):
        return self.__class__.__name__
