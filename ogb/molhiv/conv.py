import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Tanh, PReLU, ReLU, ELU, BatchNorm1d as BN
from ogb.graphproppred.mol_encoder import BondEncoder
from utils.attention import MultiheadAttention
from torch_scatter import scatter, scatter_softmax
from torch_geometric.nn.norm import MessageNorm
from torch import Tensor
from typing import Optional, List, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

class AttenConv(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(AttenConv, self).__init__(aggr='add', **kwargs)
        self.hidden = hidden

        self.proj_fea = Sequential(Linear(hidden, hidden), BN(hidden), PReLU(), Linear(hidden, hidden), PReLU())
        self.proj_mess = Sequential(Linear(hidden, hidden), BN(hidden), PReLU(), Linear(hidden, hidden), PReLU())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None
        self.attention = MultiheadAttention(hidden, 16, encode_dim=128)
        #self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=hidden)
        self.t = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True) if config.learn_t == 'true' else 1.
        learn_msg_scale = True if config.learn_msg_scale == 'true' else False
        self.msg_norm = MessageNorm(learn_msg_scale) if config.msg_norm == 'true' else None
        self.softmax_aggr = True if config.softmax_aggr == 'true' else False

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.bond_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        out = self.proj_fea(x) + self.proj_mess(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        all_feature = torch.cat((x_i.unsqueeze(0), x_j.unsqueeze(0), edge_attr.unsqueeze(0)))
        out = self.attention(all_feature)
        return torch.sum(out[0][1:],dim=0)
    
    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        if not self.softmax_aggr:
            return scatter(inputs, index, dim=-2,
                           dim_size=dim_size, reduce='add')
        else:
            out = scatter_softmax(inputs * self.t, index, dim=-2)
            return scatter(inputs * out, index, dim=-2,
                           dim_size=dim_size, reduce='add')

    def update(self, aggr_out, x):
        #all_feature = torch.cat((aggr_out.unsqueeze(0), x.unsqueeze(0)))
        #out = self.attention(all_feature, all_feature, all_feature)
        #return self.fea_mlp(torch.sum(out[0][:], dim=0))
        if self.msg_norm is not None:
            aggr_out = self.msg_norm(x, aggr_out)
        return aggr_out
        #return self.fea_mlp(aggr_out + out[0][1])

    def __repr__(self):
        return self.__class__.__name__

class AttenConv2(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(AttenConv2, self).__init__(aggr='add', **kwargs)
        self.hidden = hidden

        self.proj_fea = Sequential(Linear(hidden, hidden), BN(hidden), PReLU(), Linear(hidden, hidden), PReLU())
        self.proj_mess = Sequential(Linear(hidden, hidden), BN(hidden), PReLU(), Linear(hidden, hidden), PReLU())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

        self.attention = MultiheadAttention(hidden, 16, encode_dim=128)
        #self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=hidden)
        self.t = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True) if config.learn_t == 'true' else 1.
        learn_msg_scale = True if config.learn_msg_scale == 'true' else False
        self.msg_norm = MessageNorm(learn_msg_scale) if config.msg_norm == 'true' else None
        self.softmax_aggr = True if config.softmax_aggr == 'true' else False

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.bond_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        #out = self.fea_mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        out = self.proj_fea(x) + self.proj_mess(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        all_feature = torch.cat((x_i.unsqueeze(0), x_j.unsqueeze(0), edge_attr.unsqueeze(0)))
        out = self.attention(all_feature, all_feature, all_feature)
        return out[0][2]#torch.sum(out[0][1:],dim=0)
    
    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        if not self.softmax_aggr:
            return scatter(inputs, index, dim=-2,
                           dim_size=dim_size, reduce='sum')
        else:
            out = scatter_softmax(inputs * self.t, index, dim=-2)
            return scatter(inputs * out, index, dim=-2,
                           dim_size=dim_size, reduce='sum')

    def update(self, aggr_out, x):
        if self.msg_norm is not None:
            aggr_out = self.msg_norm(x, aggr_out)
        return aggr_out

    def __repr__(self):
        return self.__class__.__name__


class AttenConv3(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(AttenConv3, self).__init__(aggr='add', **kwargs)
        self.hidden = hidden

        self.proj_fea = Sequential(Linear(hidden, hidden), BN(hidden), PReLU(), Linear(hidden, hidden), PReLU())
        self.proj_mess = Sequential(Linear(hidden, hidden), BN(hidden), PReLU(), Linear(hidden, hidden), PReLU())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

        self.attention = MultiheadAttention(hidden, 16, encode_dim=128)
        #self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=hidden)
        self.t = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True) if config.learn_t == 'true' else 1.
        learn_msg_scale = True if config.learn_msg_scale == 'true' else False
        self.msg_norm = MessageNorm(learn_msg_scale) if config.msg_norm == 'true' else None
        self.softmax_aggr = True if config.softmax_aggr == 'true' else False

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.bond_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        #out = self.fea_mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        out = self.proj_fea(x) + self.proj_mess(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        #all_feature = torch.cat((x_j.unsqueeze(0), edge_attr.unsqueeze(0)))
        #out = self.attention(all_feature, all_feature, all_feature)
        #return torch.sum(out[0][1:],dim=0)
        return F.relu(x_j + edge_attr)
    
    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        if not self.softmax_aggr:
            return scatter(inputs, index, dim=-2,
                           dim_size=dim_size, reduce='sum')
        else:
            out = scatter_softmax(inputs * self.t, index, dim=-2)
            return scatter(inputs * out, index, dim=-2,
                           dim_size=dim_size, reduce='sum')

    def update(self, aggr_out, x):
        if self.msg_norm is not None:
            aggr_out = self.msg_norm(x, aggr_out)
        return aggr_out

    def __repr__(self):
        return self.__class__.__name__

class GinConv(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(GinConv, self).__init__(aggr='add', **kwargs)

        self.fea_mlp = Sequential(
            Linear(hidden, 2*hidden),
            BN(2*hidden),
            ReLU(),
            Linear(2*hidden, hidden))

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

        self.bond_encoder = BondEncoder(emb_dim=hidden)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.bond_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return self.__class__.__name__
