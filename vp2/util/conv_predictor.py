import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.cnv(x)


class ConvPredictor(nn.Module):
    def __init__(self, num_linear_layers=1):
        super().__init__()
        if num_linear_layers == 1:
            self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ResidualBlock(in_channels=32),
                nn.ReLU(),
                ResidualBlock(in_channels=32),
                nn.ReLU(),
                ResidualBlock(in_channels=32),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Flatten(),
                nn.Linear(12288, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ResidualBlock(in_channels=32),
                nn.ReLU(),
                ResidualBlock(in_channels=32),
                nn.ReLU(),
                ResidualBlock(in_channels=32),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Flatten(),
                nn.Linear(12288, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def forward(self, x, time_axis=False):
        if time_axis:
            shape = x.shape
            try:
                x = x.view(
                    (shape[0] * shape[1],) + shape[2:]
                )  # collapse batch*time dims [b0t0, b0t1, b0t2... b1t0, b1t1, b1t2...]
            except Exception as e:
                # if the dimensions span across subspaces, need to use reshape
                x = x.reshape(
                    (shape[0] * shape[1],) + shape[2:]
                )  # collapse batch*time dims [b0t0, b0t1, b0t2... b1t0, b1t1, b1t2...]
        x = self.net(x)
        if time_axis:
            x = x.view(
                (
                    shape[0],
                    shape[1],
                )
                + tuple(x.shape[1:])
            )
        else:
            x = x
        return x
