"""
Implementation of Stochastic Depth
"""

import torch
from torch import nn


class StochasticDepth(nn.Module):
    """
    Stochastic Depth

    Drops network paths with the given probability
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """
        Constructor Method

        Parameters
        ----------
        drop_prob : float, optional
            probability to drop the network path, by default 0.0
        """
        super().__init__()
        assert 0 <= drop_prob <= 1.0
        self.drop_prob = drop_prob

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass

        Parameters
        ----------
        X : torch.Tensor
            Input Path

        Returns:
        -------
        torch.Tensor
            Output tensor. Zeroed out with the given probability.
        """
        if torch.rand(1).item() < self.drop_prob:
            X = X * 0
        return X
