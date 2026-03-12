import torch
import torch.nn as nn


def _conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _deconv_block(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels, channel_list, out_channels):
        super().__init__()
        blocks = []
        prev = in_channels
        for c in channel_list:
            blocks.append(_conv_block(prev, c))
            prev = c
        self.conv = nn.Sequential(*blocks)
        self.out_channels = out_channels
        self.channel_list = channel_list
        self.final_c = prev

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, channel_list, out_channels):
        super().__init__()
        blocks = []
        prev = in_channels
        for c in channel_list[:-1]:
            blocks.append(_deconv_block(prev, c))
            prev = c
        blocks.append(nn.Sequential(
            nn.ConvTranspose2d(prev, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        ))
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)


class EncoderFC(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DecoderFC(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)
