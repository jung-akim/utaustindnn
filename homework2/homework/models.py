import torch
from .utils import load_data
import multiprocessing

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def set_data_dir_and_device(model, data_dir, device):
#     model.data_dir, model.device = data_dir, device

class CNNClassifier(torch.nn.Module):
    def __init__(self, layers = [16, 32], n_input_channels = 3, kernel_size = 3):
        """
        Your code here
        layers(list): number of channels per layer
        """
        data_dir = '/data/train' if device == 'cuda' else 'data/train' # Can't use arguments here to pass the grader.

        train = load_data(data_dir, shuffle=False, batch_size=None, num_workers=multiprocessing.cpu_count())
        train_data = next(iter(train))
        train_data = train_data[0].to(device)
        self.train_mean = train_data.mean(dim=[0], keepdims=True)
        self.train_std = train_data.std(dim=[0], keepdims=True)

        super().__init__()
        L = []
        # L.append(torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7))

        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size))
            L.append(torch.nn.ReLU())
            c = l
        L.append(torch.nn.MaxPool2d(kernel_size))
        L.append(torch.nn.Dropout2d(0.2))
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        # Normalization step
        x -= self.train_mean
        x /= self.train_std
        # Compute the features
        z = self.network(x)
        # Global average pooling
        z = z.mean(dim=[2, 3])
        # Classify
        return self.classifier(z)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn(test_acc = 0.838).th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn(test_acc = 0.838).th'), map_location='cpu'))
    return r
