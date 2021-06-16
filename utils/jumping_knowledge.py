import torch


class JumpingKnowledge(torch.nn.Module):

    def __init__(self, mode):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        #assert self.mode in ['cat']

    def forward(self, xs):
        assert isinstance(xs, list) or isinstance(xs, tuple)
        if self.mode != 'last':
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)
