"""
Implementation of Relative Position Bias
"""

from typing import Tuple

import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """
    Relative Postition Bias

    Inspired by timm's maxxvit implementation
    """

    def __init__(self, attn_size: Tuple[int, int], num_heads: int) -> None:
        """
        Constructor Method

        Parameters
        ----------
        attn_size : Tuple[int, int]
            Size of the attention window
        num_heads : int
            Number of heads in the multiheaded attention
        """
        super().__init__()
        self.attn_size = attn_size
        self.attn_area = self.attn_size[0] * self.attn_size[1]
        self.num_heads = num_heads

        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.attn_size[0] - 1) * (2 * self.attn_size[1] - 1),
                self.num_heads,
            )
        )

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer(
            "relative_position_index",
            self.get_relative_position_index(self.attn_size[0], self.attn_size[1]),
        )

    def get_relative_position_index(self, win_h: int, win_w: int) -> torch.Tensor:
        """
        Function to generate pair-wise relative position index for each token inside the window.

        Taken from Timms Swin V1 implementation.

        Parameters
        ----------
        win_h : int
            Window/Grid height.
        win_w : int
            Window/Grid width.

        Returns:
        -------
        torch.Tensor
            relative_coords (torch.Tensor): Pair-wise relative position indexes
            [height * width, height * width].
        """
        coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += win_h - 1
        relative_coords[:, :, 1] += win_w - 1
        relative_coords[:, :, 0] *= 2 * win_w - 1
        return relative_coords.sum(-1)

    def _get_relative_positional_bias(self) -> torch.Tensor:
        """
        Returns the relative positional bias.

        Returns:
        -------
        torch.Tensor
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self) -> torch.Tensor:
        """
        Forward Method

        Returns:
        -------
        torch.Tensor
            Pairwise relative position bias
        """
        return self._get_relative_positional_bias()
