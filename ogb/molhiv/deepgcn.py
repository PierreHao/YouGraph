import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DeepGCNLayer(torch.nn.Module):
    def __init__(self, conv=None, norm=None, act=None, block='res+',
                 dropout=0., ckpt_grad=False):
        super(DeepGCNLayer, self).__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, *args, **kwargs):
        """"""
        args = list(args)
        x = args.pop(0)

        if self.block == 'res+':
            if self.norm is not None:
                h = self.norm(x)
            else:
                h = x
            if self.act is not None:
                h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.conv is not None and self.ckpt_grad and h.requires_grad:
                h = checkpoint(self.conv, h, *args, **kwargs)
            else:
                h = self.conv(h, *args, **kwargs)

            return x + h

        else:
            if self.conv is not None and self.ckpt_grad and x.requires_grad:
                h = checkpoint(self.conv, x, *args, **kwargs)
            else:
                h = self.conv(x, *args, **kwargs)
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)

            if self.block == 'res':
                h = x + h
            elif self.block == 'dense':
                h = torch.cat([x, h], dim=-1)
            elif self.block == 'plain':
                pass

            return F.dropout(h, p=self.dropout, training=self.training)

    def __repr__(self):
        return '{}(block={})'.format(self.__class__.__name__, self.block)
