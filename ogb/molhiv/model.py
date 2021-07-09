import torch
import torch.nn.functional as F
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool, GlobalAttention
from conv import AttenConv, AttenConv2, AttenConv3, GinConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from deepgcn import DeepGCNLayer 
from gen_conv import GENConv
from utils.attention import MultiheadAttention

class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_tasks):
        super(Net, self).__init__()
        self.atom_encoder = AtomEncoder(config.hidden)
        self.bond_encoder = BondEncoder(emb_dim=config.hidden)
        self.config = config
        self.convs = torch.nn.ModuleList()
        self.beta = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        if config.methods == 'AC':
            for i in range(config.layers):
                self.convs.append(DeepGCNLayer(AttenConv(config.hidden, config.variants), block=config.block, norm=torch.nn.BatchNorm1d(config.hidden), dropout=config.dropout))
        elif config.methods == 'AC2':    
            for i in range(config.layers):
                self.convs.append(DeepGCNLayer(AttenConv2(config.hidden, config.variants), block=config.block, norm=torch.nn.BatchNorm1d(config.hidden), dropout=config.dropout))
        elif config.methods == 'AC3':    
            for i in range(config.layers):
                self.convs.append(DeepGCNLayer(AttenConv3(config.hidden, config.variants), block=config.block, norm=torch.nn.BatchNorm1d(config.hidden), dropout=config.dropout))
        elif config.methods == 'GEN':
            for i in range(config.layers):
                if config.block == 'dense':
                    self.convs.append(DeepGCNLayer(GENConv(config.hidden * (2**i), config.hidden * (2**i)), block=config.block, norm=torch.nn.BatchNorm1d(config.hidden * (2**i))))
                else:
                    self.convs.append(DeepGCNLayer(GENConv(config.hidden, config.hidden), block=config.block, norm=torch.nn.BatchNorm1d(config.hidden)))
        elif config.methods == 'GIN':
            for i in range(config.layers):
                self.convs.append(GinConv(config.hidden, config.variants))
        else:
            raise ValueError('Undefined gnn layer called {}'.format(config.methods))

        self.JK = JumpingKnowledge(config.JK)

        if config.JK == 'cat':
            self.graph_pred_linear = torch.nn.Linear(config.hidden * (config.layers + 1), num_tasks)
        else:
            if config.block == 'dense':
                self.graph_pred_linear = torch.nn.Linear(config.hidden * (2**config.layers), num_tasks)
            else:
                self.graph_pred_linear = torch.nn.Linear(config.hidden, num_tasks)

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool
        elif config.pooling == 'max':
            self.pool = global_max_pool
        elif config.pooling == "attention":
            if config.JK == 'cat':
                emb_dim = config.hidden * config.layers
            else:
                emb_dim = config.hidden
            self.pool = GlobalAttention(gate_nn = 
                    torch.nn.Sequential(
                        torch.nn.Linear(config.hidden, config.hidden)
                        ,torch.nn.BatchNorm1d(config.hidden)
                        ,torch.nn.PReLU()
                        ,torch.nn.Linear(emb_dim, 1)))


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

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, batch, mgf_maccs_pred = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch, batched_data.y[:,2]
        #edge_attr = self.bond_encoder(edge_attr)
        x = self.atom_encoder(x) + perturb if perturb is not None else self.atom_encoder(x)
        xs = [x]
        if self.config.virtual_node == 'true':
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        for i, conv in enumerate(self.convs):
            h = xs[i] + virtualnode_embedding[batch] if self.config.virtual_node == 'true' else xs[i]
            #h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index, edge_attr)
            xs.append(h)
            if self.config.virtual_node == 'true' and i < self.config.layers-1:
                virtualnode_embedding_tmp = global_add_pool(xs[i], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_tmp), p=self.dropout, training=self.training)

        nr = self.JK(xs)
        nr = F.dropout(nr, p=self.dropout, training=self.training)
        h_graph = self.pool(nr, batched_data.batch)
        #h_graph = F.dropout(h_graph, p=self.dropout, training=self.training)
        
        graph_pred =  torch.sigmoid(self.graph_pred_linear(h_graph)/self.config.T)
        h_graph_final = torch.cat((graph_pred, mgf_maccs_pred.reshape(-1,1)), 1)
        att = torch.nn.functional.softmax(h_graph_final * self.beta, -1)
        return torch.sum(h_graph_final * att, -1).reshape(-1,1)
        #return torch.clamp((1 - self.alpha)*graph_pred + self.alpha * mgf_maccs_pred.reshape(-1,1), min=0, max=1)

    def __repr__(self):
        return self.__class__.__name__

