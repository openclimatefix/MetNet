"""
Implementation of MaxViT module
"""

from dataclasses import dataclass
from typing import List, Tuple, Type, Union

import torch
from torch import nn

from metnet.layers.MBConv import MBConv
from metnet.layers.PartitionAttention import BlockAttention, GridAttention


@dataclass
class MaxViTDataClass:
    """
    DataClass for MaxViT

    Parameters
    ----------
    mb_conv_expansion_rate : int, optional
        MBConv: Expansion rate for the output channels, by default 4
    mb_conv_downscale : bool, optional
        MBConv: Flag to denote downscaling in the conv branch, by default False
    mb_conv_act_layer : Type[nn.Module], optional
        MBConv: activation layer, by default nn.GELU
    mb_conv_norm_layer : Type[nn.Module], optional
        MBConv: norm layer, by default nn.BatchNorm2D
    mb_conv_drop_path : float, optional
        MBConv: Stochastic Depth ratio, by default 0.0
    mb_conv_kernel_size : int, optional
        MBConv: Conv kernel size, by default 3
    mb_conv_se_bottleneck_ratio : float, optional
        MBConv: Squeeze Excite reduction ratio, by default 0.25
    block_attention_num_heads : int, optional
        BlockAttention: Number of attention heads, by default 32
    block_attention_channels : int
        BlockAttention: Number of channels used for attention computations.
        It should be divisible by num_heads, by default 64
    block_attention_attn_grid_window_size : Tuple[int, int], optional
        BlockAttention: Grid/Window size for attention, by default (8, 8)
    block_attention_attn_drop : float, optional
        BlockAttention: Dropout ratio of attention weight, by default 0
    block_attention_proj_drop : float, optional
        BlockAttention: Dropout ratio of output, by default 0
    block_attention_drop_path : float, optional
        BlockAttention: Stochastic depth, by default 0
    block_attention_pre_norm_layer : Type[nn.Module], optional
        BlockAttention: Pre norm layer, by default nn.LayerNorm
    block_attention_post_norm_layer : Type[nn.Module], optional
        BlockAttention: Post norm layer, by default nn.LayerNorm
    block_attention_use_mlp : Type[nn.Module], optional
        BlockAttention: MLP to be used after the attention, by default None
    block_attention_use_normalised_qk : bool, optional
        BlockAttention: Normalise queries and keys as done in Metnet 3, by default True
    grid_attention_num_heads : int, optional
        GridAttention: Number of attention heads, by default 32
    grid_attention_channels : int
        GridAttention: Number of channels used for attention computations.
        It should be divisible by num_heads, by default 64
    grid_attention_attn_grid_window_size : Tuple[int, int], optional
        GridAttention: Grid/Window size for attention, by default (8, 8)
    grid_attention_attn_drop : float, optional
        GridAttention: Dropout ratio of attention weight, by default 0
    grid_attention_proj_drop : float, optional
        GridAttention: Dropout ratio of output, by default 0
    grid_attention_drop_path : float, optional
        GridAttention: Stochastic depth, by default 0
    grid_attention_pre_norm_layer : Type[nn.Module], optional
        GridAttention: Pre norm layer, by default nn.LayerNorm
    grid_attention_post_norm_layer : Type[nn.Module], optional
        GridAttention: Post norm layer, by default nn.LayerNorm
    grid_attention_use_mlp : Type[nn.Module], optional
        GridAttention: MLP to be used after the attention, by default None
    grid_attention_use_normalised_qk : bool, optional
        GridAttention: Normalise queries and keys as done in Metnet 3, by default True
    """

    mb_conv_expansion_rate: int = 4
    mb_conv_downscale: bool = False
    mb_conv_act_layer: Type[nn.Module] = nn.GELU
    mb_conv_norm_layer: Type[nn.Module] = nn.BatchNorm2d
    mb_conv_drop_path: float = 0.0
    mb_conv_kernel_size: int = 3
    mb_conv_se_bottleneck_ratio: float = 0.25
    block_attention_num_heads: int = 32
    block_attention_channels: int = 64
    block_attention_attn_grid_window_size: Tuple[int, int] = (8, 8)
    block_attention_attn_drop: float = 0
    block_attention_proj_drop: float = 0
    block_attention_drop_path: float = 0
    block_attention_pre_norm_layer: Type[nn.Module] = nn.LayerNorm
    block_attention_post_norm_layer: Type[nn.Module] = nn.LayerNorm
    block_attention_use_mlp: Type[nn.Module] = None
    block_attention_use_normalised_qk: bool = True
    grid_attention_num_heads: int = 32
    grid_attention_channels: int = 64
    grid_attention_attn_grid_window_size: Tuple[int, int] = (8, 8)
    grid_attention_attn_drop: float = 0
    grid_attention_proj_drop: float = 0
    grid_attention_drop_path: float = 0
    grid_attention_pre_norm_layer: Type[nn.Module] = nn.LayerNorm
    grid_attention_post_norm_layer: Type[nn.Module] = nn.LayerNorm
    grid_attention_use_mlp: Type[nn.Module] = None
    grid_attention_use_normalised_qk: bool = True


class MaxViTBlock(nn.Module):
    """
    MaxViT block
    """

    def __init__(self, in_channels: int, maxvit_config: Type[MaxViTDataClass]) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of input channels
        maxvit_config : Type[MaxViTDataClass]
            MaxVit Config
        """
        super().__init__()
        self.in_channels = in_channels
        self.config = maxvit_config

        self.mb_conv = MBConv(
            in_channels=self.in_channels,
            expansion_rate=self.config.mb_conv_expansion_rate,
            downscale=self.config.mb_conv_downscale,
            act_layer=self.config.mb_conv_act_layer,
            norm_layer=self.config.mb_conv_norm_layer,
            drop_path=self.config.mb_conv_drop_path,
            kernel_size=self.config.mb_conv_kernel_size,
            se_bottleneck_ratio=self.config.mb_conv_se_bottleneck_ratio,
        )

        # Init Block and Grid Attention

        self.block_attention = BlockAttention(
            in_channels=self.in_channels,
            num_heads=self.config.block_attention_num_heads,
            attention_channels=self.config.block_attention_channels,
            attn_grid_window_size=self.config.block_attention_attn_grid_window_size,
            attn_drop=self.config.block_attention_attn_drop,
            proj_drop=self.config.block_attention_proj_drop,
            drop_path=self.config.block_attention_drop_path,
            pre_norm_layer=self.config.block_attention_pre_norm_layer,
            post_norm_layer=self.config.block_attention_post_norm_layer,
            use_mlp=self.config.block_attention_use_mlp,
            use_normalised_qk=self.config.block_attention_use_normalised_qk,
        )

        self.grid_attention = GridAttention(
            in_channels=self.in_channels,
            num_heads=self.config.grid_attention_num_heads,
            attention_channels=self.config.grid_attention_channels,
            attn_grid_window_size=self.config.grid_attention_attn_grid_window_size,
            attn_drop=self.config.grid_attention_attn_drop,
            proj_drop=self.config.grid_attention_proj_drop,
            drop_path=self.config.grid_attention_drop_path,
            pre_norm_layer=self.config.grid_attention_pre_norm_layer,
            post_norm_layer=self.config.grid_attention_post_norm_layer,
            use_mlp=self.config.grid_attention_use_mlp,
            use_normalised_qk=self.config.grid_attention_use_normalised_qk,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of the shape [N, C, H, W]

        Returns:
        -------
        torch.Tensor
            MaxViT block output tensor of the shape [N, C, H, W]
        """

        output = self.mb_conv(X)
        output = self.block_attention(output)
        output = self.grid_attention(output)

        return output


class MetNetMaxVit(nn.Module):
    """
    MaxViT block implemented with MetNet 3 modifications.
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = None,
        num_blocks: int = 12,
        maxvit_conf: Union[Type[MaxViTDataClass], List[Type[MaxViTDataClass]]] = MaxViTDataClass(),
        set_linear_stocastic_depth: bool = True,
    ) -> None:
        """
        MetNet3 MaxViT Block

        Parameters
        ----------
        in_channels : int, optional
            Input Channels, by default 512
        out_channels : int, optional
            Output Channels, by default None
        num_blocks : int, optional
            Number of MaxViT blocks, by default 12
        maxvit_conf : Union[ Type[MaxViTDataClass], List[Type[MaxViTDataClass]] ], optional
            MaxViT config, by default MaxViTDataClass()
        set_linear_stocastic_depth : bool, optional
            Flag to set the stochastic depth linearly in each MaxVit subblock, by default True
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.num_blocks = num_blocks
        self.set_linear_stocastic_depth = set_linear_stocastic_depth

        self.maxvit_blocks = nn.ModuleList()

        if isinstance(maxvit_conf, List):
            assert len(maxvit_conf) == num_blocks
            self.maxvit_conf_list = maxvit_conf
        else:
            self.maxvit_conf_list = [maxvit_conf for _ in range(self.num_blocks)]

        if self.set_linear_stocastic_depth:
            # Linearly sets the stochastic depth a given sub-module
            # (i.e. MBConv, local (block) attention or gridded (grid) attention)
            # from 0 to 0.2, as mentioned in Metnet3 paper
            for conf in self.maxvit_conf_list:
                conf.mb_conv_drop_path = 0
                conf.block_attention_drop_path = 0.1
                conf.grid_attention_drop_path = 0.2

        for conf in self.maxvit_conf_list:
            self.maxvit_blocks.append(MaxViTBlock(in_channels=self.in_channels, maxvit_config=conf))

        self.linear_transform = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        Parameters
        ----------
        X : torch.Tensor
            Input tensor

        Returns:
        -------
        torch.Tensor
            Output of the MaxViT block.
        """
        model_output_list = []
        model_output_list.append(self.maxvit_blocks[0](X))
        for i in range(1, self.num_blocks):
            model_output_list.append(self.maxvit_blocks[i](model_output_list[i - 1]))

        output = X + torch.stack(model_output_list).sum(dim=0)

        output = self.linear_transform(output)
        return output
