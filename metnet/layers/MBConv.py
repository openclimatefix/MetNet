"""
MBConv Implementation
"""

from typing import Type

import torch
from torch import nn

from metnet.layers.SqueezeExcitation import SqueezeExcite
from metnet.layers.StochasticDepth import StochasticDepth


class MBConv(nn.Module):
    """
    MB Conv implementation, inspired by timm's implementation.
    """

    def __init__(
        self,
        in_channels: int,
        expansion_rate: int = 4,
        downscale: bool = False,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        se_bottleneck_ratio: float = 0.25,
    ):
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Input channels
        expansion_rate : int, optional
            Expansion rate for the output channels, by default 4
        downscale : bool, optional
            Flag to denote downscaling in the conv branch, by default False
            Currently not implemented, as not specified in Metnet 3
        act_layer : Type[nn.Module], optional
            activation layer, by default nn.GELU
        norm_layer : Type[nn.Module], optional
            normalisation layer, by default nn.BatchNorm2d
            TODO: Verify if Layer Norm is to to be used inside MBConv
            NOTE: Most implementations use nn.BatchNorm2d. If LayerNorm is to be
            used, the intermediate h and w would need to be computed.
        drop_path : float, optional
            Stochastic Depth ratio, by default 0.0
        kernel_size : int, optional
            Conv kernel size, by default 3
        se_bottleneck_ratio : float, optional
            Squeeze Excite reduction ratio, by default 0.25
        """
        # TODO: Verify implemtnetation
        super().__init__()
        self.in_channels = in_channels
        self.drop_path_rate = drop_path
        self.expansion_rate = expansion_rate
        self.downscale = downscale

        out_channels = self.in_channels * self.expansion_rate
        conv_dw_padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)

        self.pre_norm = norm_layer(in_channels)
        self.norm_layer_1 = norm_layer(out_channels)
        self.norm_layer_2 = norm_layer(out_channels)

        if self.downscale:
            # TODO: Check if downscaling is needed at all. May impact layer normalisation.
            raise NotImplementedError(
                "Downscaling in MBConv hasn't been implemented as it \
                isnt used in Metnet3"
            )

        self.main_branch = nn.Sequential(
            self.pre_norm,  # Pre Normalize over the last three dimensions (i.e. the channel and spatial dimensions) # noqa
            nn.Conv2d(  # Conv 1x1
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            self.norm_layer_1,  # Norm1
            nn.Conv2d(  # Depth wise Conv kxk
                out_channels,
                out_channels,
                kernel_size,
                padding=conv_dw_padding,  # To maintain shapes
                groups=out_channels,
            ),
            self.norm_layer_2,  # Norm2
            SqueezeExcite(
                in_channels=out_channels,
                act_layer=act_layer,
                rd_ratio=se_bottleneck_ratio,
            ),
            nn.Conv2d(  # Conv 1x1
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            # No Norm as this is the last convolution layer in this block
        )

        self.stochastic_depth = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward step

        Parameters
        ----------
        X : torch.Tensor
            Input Tensor of shape [N, C, H, W]

        Returns:
        -------
        torch.Tensor
            MBConv output
        """
        return X + self.stochastic_depth(self.main_branch(X))
