"""
MultiHeaded 2D Self Attention  Implementation
"""

from typing import Type

import torch
import torch.nn as nn


class MultiheadSelfAttention2D(nn.Module):
    """
    Implementing multi-head self-attention for 2D images.
    """

    def __init__(
        self,
        in_channels: int,
        attention_channels: int = 64,
        num_heads: int = 16,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_normalised_qk: bool = True,
        rel_attn_bias: Type[nn.Module] = None,
    ) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        attention_channels : int
            Number of channels used for attention computations.
            It should be divisible by num_heads, by default 64
        num_heads : int
            Number of attention heads, by default 16
        attn_drop : float, optional
            attention dropout rate, by default 0.0
        proj_drop : float, optional
            post attention projection dropout rate, by default 0.0
        use_normalised_qk : bool, by default True
            Normalise queries and keys, (as in Metnet 3)
        rel_attn_bias : Type[nn.Module], optional
            Use Relative Position bias, by default None
        """
        super().__init__()

        assert (
            attention_channels % num_heads == 0
        ), "attention_channels should be divisible by num_heads"

        self.in_channels: int = in_channels
        self.num_heads = num_heads
        self.attention_head_size = attention_channels // num_heads

        # Use relative Position Bias
        self.rel_attn_bias = rel_attn_bias

        # Linear transformations for Q, K and V
        self.query = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, attention_channels, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv2d(attention_channels, in_channels, kernel_size=1)

        # Normalised Keys and Queries as specified in Metnet 3
        self.use_normalised_qk = use_normalised_qk

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the multi-head self-attention mechanism.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns:
        -------
        torch.Tensor
            Output tensor after multi-head self-attention of shape (N, C, H, W).
        """

        N, C, H, W = X.size()
        # Compute Q, K, V
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        Q = Q.view(N, self.num_heads, self.attention_head_size, H * W)
        K = K.view(N, self.num_heads, self.attention_head_size, H * W)
        V = V.view(N, self.num_heads, self.attention_head_size, H * W)

        if self.use_normalised_qk:
            Q = torch.nn.functional.normalize(Q, dim=-2)  # TODO: verify dim
            K = torch.nn.functional.normalize(K, dim=-2)  # TODO: verify dim

        Q = Q.transpose(
            -1, -2
        )  # Making Q of shape [N, self.num_heads, H*W, self.attention_head_size]

        attention_weights = Q @ K  # Attn shape [N, self.num_heads, H*W, H*W]

        if self.rel_attn_bias is not None:
            attention_weights = attention_weights + self.rel_attn_bias()

        attention_weights = attention_weights.softmax(
            dim=-1
        )  # Attn shape [N, self.num_heads, H*W H*W]

        # Multiply attention weights with V
        # V shape [N, self.num_heads, self.attention_head_size, H*W]
        # Attn shape [N, self.num_heads, H*W, H*W]

        attention_weights = self.attn_drop(attention_weights)
        out = V @ attention_weights  # Out shape [N, self.num_heads, H*W, self.attention_head_size]

        # Combine the heads
        out = out.permute(
            0, 1, 3, 2
        )  # Out shape [N, self.num_heads, self.attention_head_size, H*W]
        out = out.contiguous().view(N, self.num_heads * self.attention_head_size, H, W)

        # Apply the output projection
        # Out shape [N, attention_channels, H, W]
        out = self.out_proj(out)  # Out shape [N, C, H, W]
        out = self.proj_drop(out)
        return out + X
