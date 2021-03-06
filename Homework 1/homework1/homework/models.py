import torch
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        # return (-torch.log(input.exp() / input.exp().sum(axis = 1).view(-1,1))[range(input.size(0)), target.to(torch.long)]).mean()
        return torch.nn.functional.cross_entropy(input, target.to(torch.long))

        # raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        # self.w = Parameter(torch.zeros(input_dim))
        # self.b = Parameter(torch.zeros(1))

        self.linear = torch.nn.Linear(64*64 *3, 6)

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        # logit = (x * self.w[None, :]).sum(dim = 1) + self.b

        # self.linear = torch.nn.Linear(x.view(-1), 6)
        # logit = self.linear(x)
        # print(logit.shape)
        return self.linear(x.view(x.shape[0], -1))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        n_neurons = 10
        self.linear = torch.nn.Linear(64 * 64 * 3, n_neurons)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n_neurons, 6)

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        o = self.linear(x.view(x.shape[0], -1))
        o = self.relu(o)
        o = self.linear2(o)
        return self.relu(o)


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
