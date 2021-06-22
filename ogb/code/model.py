import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, PReLU, ReLU, ELU, Sigmoid
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool
from conv import AttenConv, GinConv, ExpC, CombC, ExpC_star, CombC_star


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_vocab, max_seq_len, node_encoder):
        super(Net, self).__init__()
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len

        self.node_encoder = node_encoder

        self.convs = torch.nn.ModuleList()
        self.config = config
        if config.methods == 'AC':
            for i in range(config.layers):
                self.convs.append(AttenConv(config.hidden, config.variants))
        elif config.methods == 'GIN':
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

        self.graph_pred_linear_list = torch.nn.ModuleList()
        if config.JK == 'cat':
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(config.hidden * (config.layers + 1), self.num_vocab))
        else:
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(config.hidden, self.num_vocab))

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool

        self.dropout = config.dropout
        
        # virtualnode
        if self.config.virtual_node == 'true':
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

    def forward(self, batched_data):
        '''
            Return:
                A list of predictions.
                i-th element represents prediction at i-th position of the sequence.
        '''
        x, edge_index, edge_attr, node_depth, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.node_depth, batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1,))
        xs = [x]
        if self.config.virtual_node == 'true':
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        for i, conv in enumerate(self.convs):
            x = xs[i] + virtualnode_embedding[batch] if self.config.virtual_node == 'true' else xs[i]
            x = conv(x, edge_index, edge_attr)
            xs += [x]
            if self.config.virtual_node == 'true' and i < self.config.layers-1:
                virtualnode_embedding_tmp = global_add_pool(xs[i], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_tmp), p=self.dropout, training=self.training)

        nr = self.JK(xs)

        nr = F.dropout(nr, p=self.dropout, training=self.training)
        h_graph = self.pool(nr, batched_data.batch)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph) / self.config.T)

        return pred_list

    def __repr__(self):
        return self.__class__.__name__
