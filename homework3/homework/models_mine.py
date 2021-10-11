import torch
import torch.nn.functional as F
from homework.utils import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAIN_MEAN, TRAIN_STD = 0., 1.

class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride = 1, resnet = False):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )

            self.downsample = None
            if resnet:
                # Making the same (n_input, n_output) shape as input for residual network
                if stride != 1 or n_input != n_output:
                    self.downsample = torch.nn.Sequential(torch.nn.Dropout2d(0.2),
                                                          torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                                                          torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            if self.downsample is not None:
                identity = x
                identity = self.downsample(identity)
                return self.net(x) + identity
            else:
                return self.net(x)

    def __init__(self, layers = [16, 32, 64], n_input_channels = 3):

        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """

        c = 32
        L = [torch.nn.Conv2d(n_input_channels, c, kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(c),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        for l in layers:
            L.append(self.Block(c, l, stride=2, resnet=True))
            c = l

        L.append(torch.nn.Dropout2d(0.2))
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        z = self.network(x)
        z = z.mean(dim = [2,3])

        # return self.classifier(z)[:, 0]
        return self.classifier(z)

class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride = 2, kernel_size = 3, padding=1, maxpool = False):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=padding, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=padding),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )

            self.downsample = None

            # Making the same (n_input, n_output) shape as input for residual network
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Dropout2d(0.2),
                                                      torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                                                      torch.nn.BatchNorm2d(n_output),
                                                      torch.nn.ReLU())
            if maxpool:
                self.net.add_module('maxpool2d', torch.nn.MaxPool2d(kernel_size=2))
                self.downsample.add_module('maxpool2d', torch.nn.MaxPool2d(kernel_size=2))

        def forward(self, x):
            if self.downsample is not None:
                return self.net(x) + self.downsample(x)
            return self.net(x)

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size = 3, stride = 2, padding = 1, output_padding = 1):
            super().__init__()
            k, s, p, o = kernel_size, stride, padding, output_padding
            self.net = torch.nn.Sequential(torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=k, stride=s, padding=p, output_padding=o),
                                           torch.nn.BatchNorm2d(n_output),
                                           torch.nn.ReLU())
        def forward(self, x):
            return self.net(x)

    def __init__(self, layers = [16, 32, 64, 128], stride = 2, kernel_size = 3, n_input_channels = 3):

        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        # data_dir = '/dense_data/train' if device == 'cuda' else 'dense_data/train' # Can't use arguments here to pass the grader.
        # train = load_dense_data(data_dir, batch_size=10000, shuffle=False, num_workers=0)
        # train_data = next(iter(train))

        try:
            train_data = torch.load('train_data.pt')
        except Exception:
            train_data = torch.load('homework/train_data.pt')
        except:
            print('train_data.pt doesn\'t exist.')

        train_data = train_data[0].to(device)

        global TRAIN_MEAN
        TRAIN_MEAN = train_data.mean(dim=[0,2,3], keepdims=True)
        global TRAIN_STD
        TRAIN_STD = train_data.std(dim=[0,2,3], keepdims=True)

        NUM_CLASSES = 5
        c = n_input_channels
        L = []

        # C, W, H
        # 3   -> 16 -> 32 -> 64 -> 64  -> 128  -> 64 -> 32 -> 16  -> 8  -> NUM_CLASSES
        # 96  -> 48 -> 24 -> 12 -> 6   -> 3    -> 6  -> 12 -> 24 -> 48  -> 96
        # 128 -> 64 -> 32 -> 16 -> 8   -> 4    -> 8  -> 16 -> 32 -> 64  -> 128
        #        x1    x2    x3    x4     x5      u1    u2    u3    u4     u5
        s, p, k = stride, stride // 2, kernel_size

        self.net1 = self.Block(n_input_channels, layers[0], stride=s, padding=p, kernel_size=k)
        self.down4 = torch.nn.Conv2d(layers[0], 8, kernel_size=1, stride=1)

        self.net2 = self.Block(layers[0], layers[1], stride=s, padding=p, kernel_size=k)
        self.down3 = torch.nn.Conv2d(layers[1], 16, kernel_size=1, stride = 1)

        self.net3 = self.Block(layers[1], layers[2], stride=s, padding=p, kernel_size=k)
        self.down2 = torch.nn.Conv2d(layers[2], layers[1], kernel_size=1, stride = 1)

        self.downpool = torch.nn.MaxPool2d(kernel_size=2)
        self.net_except = self.Block(layers[2], layers[2], stride=s, padding=p, kernel_size=k)
        self.down1 = torch.nn.Conv2d(layers[2], layers[2], kernel_size=1, stride = 1)

        self.net4 = self.Block(layers[2], layers[3], stride=s, padding=p, kernel_size=k)

        self.upconv1 = self.UpBlock(layers[3], layers[2], kernel_size=k, stride=s, padding=p, output_padding=1)
        self.upconv2 = self.UpBlock(layers[2]+layers[2], layers[1], kernel_size=k, stride=s, padding=p, output_padding=1)
        self.upconv3 = self.UpBlock(layers[1]+layers[1], layers[0], kernel_size=k, stride=s, padding=p, output_padding=1)
        self.upconv4 = self.UpBlock(layers[0]+layers[0], 8, kernel_size=k, stride=s, padding=p, output_padding=1)
        self.upconv5 = self.UpBlock(8+8, NUM_CLASSES, kernel_size=k, stride=s, padding=p, output_padding=1)

        self.dropout = torch.nn.Dropout2d(0.2)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        if x.shape[1:] == TRAIN_MEAN.shape[1:]:
            x -= TRAIN_MEAN
            x /= TRAIN_STD

        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        try:
            x4 = self.maxpool(x3)
        except:
            x4 = self.net_except(x3)
        x5 = self.net4(x4)

        u1 = self.upconv1(x5)[:, :, :x4.shape[2], :x4.shape[3]] # Cut off potential over-padding caused by output_padding
        u1 = torch.cat((u1, self.down1(x4)), 1)

        u2 = self.upconv2(u1)[:, :, :x3.shape[2], :x3.shape[3]]
        u2 = torch.cat((u2, self.down2(x3)), 1)

        u3 = self.upconv3(u2)[:, :, :x2.shape[2], :x2.shape[3]]
        u3 = torch.cat((u3, self.down3(x2)), 1)

        u4 = self.upconv4(u3)[:, :, :x1.shape[2], :x1.shape[3]]
        u4 = torch.cat((u4, self.down4(x1)), 1)

        u5 = self.upconv5(u4)[:, :, :x.shape[2], :x.shape[3]]

        return u5

model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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

if __name__ == '__main__':
    model = FCN()
    # x = torch.rand(128, 3, 96, 128)
    # x = torch.rand(1, 3, 1, 1)
    x = torch.rand(1, 3, 32, 1)
    model.eval()
    y = model(x)

    print(y.shape)