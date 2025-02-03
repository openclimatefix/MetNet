"""Dilated Time Conditioned Residual Convolution Block for MetNet-2"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from metnet.layers.LeadTimeConditioner import LeadTimeConditioner


class DilatedResidualConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 384,
        dilation: int = 1,
        kernel_size: int = 3,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.output_channels = output_channels
        self.dilated_conv_one = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            dilation=(dilation, dilation),
            kernel_size=(kernel_size, kernel_size),
            padding="same",
        )
        # Target Time index conditioning
        self.lead_time_conditioner = LeadTimeConditioner()
        self.activation = activation
        self.dilated_conv_two = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            dilation=(dilation, dilation),
            kernel_size=(kernel_size, kernel_size),
            padding="same",
        )
        # To make sure number of channels match, might need a 1x1 conv
        if input_channels != output_channels:
            self.channel_changer = nn.Conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=(1, 1)
            )
        else:
            self.channel_changer = nn.Identity()

    def forward(self, x: torch.Tensor, beta, gamma) -> torch.Tensor:
        out = self.dilated_conv_one(x)
        out = F.layer_norm(out, out.size()[1:])
        out = self.lead_time_conditioner(out, beta, gamma)
        out = self.activation(out)
        out = self.dilated_conv_two(out)
        out = F.layer_norm(out, out.size()[1:])
        out = self.lead_time_conditioner(out, beta, gamma)
        out = self.activation(out)
        x = self.channel_changer(x)
        return x + out


class UpsampleResidualConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 512,
        dilation: int = 1,
        kernel_size: int = 3,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.output_channels = output_channels
        self.dilated_conv_one = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels,
            stride=2,
            kernel_size=kernel_size,
        )
        # Target Time index conditioning
        self.lead_time_conditioner = LeadTimeConditioner()
        self.activation = activation
        self.dilated_conv_two = nn.ConvTranspose2d(
            in_channels=output_channels,
            out_channels=output_channels,
            stride=2,
            kernel_size=kernel_size,
        )

        if input_channels != output_channels:
            self.channel_changer = nn.Conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=(1, 1)
            )
        else:
            self.channel_changer = nn.Identity()

    def forward(self, x: torch.Tensor, beta, gamma) -> torch.Tensor:
        out = self.dilated_conv_one(x)
        out = F.layer_norm(out, out.size()[1:])
        out = self.lead_time_conditioner(out, beta, gamma)
        out = self.activation(out)
        out = self.dilated_conv_two(out)
        out = F.layer_norm(out, out.size()[1:])
        out = self.lead_time_conditioner(out, beta, gamma)
        out = self.activation(out)
        x = self.channel_changer(x)
        return x + out
