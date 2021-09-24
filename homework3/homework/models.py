import torch
import torch.nn.functional as F


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
        def __init__(self, n_input, n_output, stride = 2, kernel_size = 3, padding=1):
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
        def forward(self, x):
            if self.downsample is not None:
                return self.net(x) + self.downsample(x)
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
        NUM_CLASSES = 5
        c = n_input_channels
        L = []

        # Regular NN
        # 3   -> 16 -> 32 -> 64 -> 128 -> 64 + 64 -> 32 + 32 -> 16 + 16 -> NUM_CLASSES
        # 96  -> 48 -> 24 -> 12 -> 6   -> 12      -> 24      -> 48      -> 96
        # 128 -> 64 -> 32 -> 16 -> 8   -> 16      -> 32      -> 64      -> 128

        s, p, k = stride, stride//2, kernel_size

        self.net1 = self.Block(n_input_channels, layers[0], stride=s, padding=p, kernel_size=k)

        self.net2 = self.Block(layers[0], layers[1], stride=s, padding=p, kernel_size=k)
        self.down3 = torch.nn.Conv2d(layers[0], layers[0], kernel_size=1, stride = 1)

        self.net3 = self.Block(layers[1], layers[2], stride=s, padding=p, kernel_size=k)
        self.down2 = torch.nn.Conv2d(layers[1], layers[1], kernel_size=1, stride = 1)

        self.net4 = self.Block(layers[2], layers[3], stride = s, padding = p, kernel_size=k)
        self.down1 = torch.nn.Conv2d(layers[2], layers[2], kernel_size=1, stride = 1)

        self.net5 = torch.nn.ConvTranspose2d(layers[3], layers[2], kernel_size=k, stride=s, padding=p, output_padding=1)
        self.net6 = torch.nn.ConvTranspose2d(layers[2] + layers[2], layers[1], kernel_size=k, stride=s, padding=p, output_padding=1)
        self.net7 = torch.nn.ConvTranspose2d(layers[1] + layers[1], layers[0], kernel_size=k, stride=s, padding=p, output_padding=1)
        self.net8 = torch.nn.ConvTranspose2d(layers[0] + layers[0], NUM_CLASSES, kernel_size=k, stride=s, padding=p, output_padding=1)

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
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        x4 = self.net4(x3)

        x5 = self.net5(x4)[:, :, :x3.shape[2], :x3.shape[3]] # Cut off potential over-padding caused by output_padding
        x5 = torch.cat((x5, self.down1(x3)), 1)

        x6 = self.net6(x5)[:, :, :x2.shape[2], :x2.shape[3]]
        x6 = torch.cat((x6, self.down2(x2)), 1)

        x7 = self.net7(x6)[:, :, :x1.shape[2], :x1.shape[3]]
        x7 = torch.cat((x7, self.down3(x1)), 1)

        x8 = self.net8(x7)[:, :, :x.shape[2], :x.shape[3]]

        return x8

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
    x = torch.rand(128, 3, 96, 128)
    # x = torch.rand(1, 3, 1, 1)
    model.eval()
    y = model(x)

    print(y.shape)