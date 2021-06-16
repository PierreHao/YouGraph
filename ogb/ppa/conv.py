import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn.functional as F
from torch.nn import PReLU, Linear, Sequential, Tanh, ReLU, ELU, BatchNorm1d as BN

class ExpC(MessagePassing):
    def __init__(self, hidden, num_aggr, config, **kwargs):
        super(ExpC, self).__init__(aggr='add', **kwargs)
        self.hidden = hidden
        self.num_aggr = num_aggr
        
        self.fea_mlp = Sequential(
            Linear(hidden * self.num_aggr, hidden),
            PReLU(),
            Linear(hidden, hidden)
            ,PReLU())

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, self.num_aggr),
            Tanh())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

        self.edge_encoder = torch.nn.Linear(7, hidden)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        feature2d = torch.matmul(aggr_emb.unsqueeze(-1), xe.unsqueeze(-1)
                                 .transpose(-1, -2)).squeeze(-1).view(-1, self.hidden * self.num_aggr)
        return self.fea_mlp(feature2d)

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        feature2d = torch.matmul(root_emb.unsqueeze(-1), x.unsqueeze(-1)
                                 .transpose(-1, -2)).squeeze(-1).view(-1, self.hidden * self.num_aggr)
        return aggr_out + self.fea_mlp(feature2d)

    def __repr__(self):
        return self.__class__.__name__


class ExpC_star(MessagePassing):
    def __init__(self, hidden, num_aggr, config, **kwargs):
        super(ExpC_star, self).__init__(aggr='add', **kwargs)
        self.hidden = hidden
        self.num_aggr = num_aggr

        self.fea_mlp = Sequential(
            Linear(hidden * self.num_aggr, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU())

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, self.num_aggr),
            Tanh())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

        self.edge_encoder = torch.nn.Linear(7, hidden)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        feature2d = torch.matmul(aggr_emb.unsqueeze(-1), xe.unsqueeze(-1)
                                 .transpose(-1, -2)).squeeze(-1).view(-1, self.hidden * self.num_aggr)
        return feature2d

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        feature2d = torch.matmul(root_emb.unsqueeze(-1), x.unsqueeze(-1)
                                 .transpose(-1, -2)).squeeze(-1).view(-1, self.hidden * self.num_aggr)
        return aggr_out + feature2d

    def __repr__(self):
        return self.__class__.__name__


class CombC(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(CombC, self).__init__(aggr='add', **kwargs)

        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU())

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, hidden),
            Tanh())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

        self.edge_encoder = torch.nn.Linear(7, hidden)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        return self.fea_mlp(aggr_emb * xe)

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        return aggr_out + self.fea_mlp(root_emb * x)

    def __repr__(self):
        return self.__class__.__name__


class CombC_star(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(CombC_star, self).__init__(aggr='add', **kwargs)

        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU())

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, hidden),
            Tanh())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

        self.edge_encoder = torch.nn.Linear(7, hidden)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        return aggr_emb * xe

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        return aggr_out + root_emb * x

    def __repr__(self):
        return self.__class__.__name__


class GinConv(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(GinConv, self).__init__(aggr='add', **kwargs)

        self.fea_mlp = Sequential(
            Linear(hidden, 2*hidden),
            BN(hidden*2),
            PReLU(),
            Linear(2*hidden, hidden))

        if config.BN == 'Y':
            self.BN = BN(hidden)
            self.act = PReLU()
        else:
            self.BN = None

        self.edge_encoder = torch.nn.Linear(7, hidden)
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        #self.fea_act = PReLU()

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        #out = self.fea_mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        out = self.proj_fea(x) + self.proj_mess(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_j, edge_attr):
        all_feature = torch.cat((x_i.unsqueeze(0), x_j.unsqueeze(0), edge_attr.unsqueeze(0)))
        out = self.attention(all_feature, all_feature, all_feature)
        return torch.sum(out[0][1:],dim=0)

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return self.__class__.__name__
