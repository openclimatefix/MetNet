"""
Squeeze-and-Excitation net implementation
"""

from typing import Type

import torch
from torch import nn


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation net
    """

    def __init__(
        self,
        in_channels: int,
        rd_ratio: float = 0.25,
        act_layer: Type[nn.Module] = nn.ReLU,
        gate_layer: Type[nn.Module] = nn.Sigmoid,
    ):
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            input channels to layer
        rd_ratio : float, optional
            ratio of squeeze reduction, by default 0.25
        act_layer : Type[nn.Module], optional
            activation layer of containing block, by default nn.ReLU
        gate_layer : Type[nn.Module], optional
            attention gate function, by default nn.Sigmoid
        """
        super().__init__()
        self.in_channels = in_channels
        self.rd_ratio = rd_ratio
        self.out_channels = round(self.in_channels * self.rd_ratio)
        self.conv_reduce = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(self.out_channels, self.in_channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass

        Parameters
        ----------
        X : torch.Tensor
            Input Tensor

        Returns:
        -------
        torch.Tensor
            Output Tensor
        """
        x_se = X.mean((2, 3), keepdim=True)  # Mean along H, W dim
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return X * self.gate(x_se)
