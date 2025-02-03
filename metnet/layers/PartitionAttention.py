"""
Implementation of Partition (Grid and Block) Attention
"""

from typing import Tuple, Type

import torch
from torch import Tensor, nn

from metnet.layers.MultiheadSelfAttention2D import MultiheadSelfAttention2D
from metnet.layers.RelativePositionBias import RelativePositionBias
from metnet.layers.StochasticDepth import StochasticDepth


class PointwiseMLP(nn.Module):
    """
    Pointwise MLP for [N, C, H, W] images
    """

    def __init__(self, in_channels: int, out_channels: int = None, hidden_dim: int = None) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Input Channels
        out_channels : int, optional
            Output Channels, by default None
            If None, set equal to in_channels
        hidden_dim : int, optional
            Hidden Dim for MLP, by default None
            If None, set equal to 4 * in_channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.hidden_dim = hidden_dim if hidden_dim else 4 * self.in_channels

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_channels),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward Method

        Parameters
        ----------
        X : torch.Tensor
            Input Tensor

        Returns:
        -------
        torch.Tensor
            Output Tensor
        """
        X = X.permute(0, 2, 3, 1)  # Converting from [N, C, H, W] to [N, W, C, W]
        return self.mlp(X)


class PartitionAttention(nn.Module):
    """
    Partition Attention

    Implements the common functionality for block and grid attention.
    X ← X + StochasticDepth(RelAttention(PreNorm(Partition(X))))
    X ← X + StochasticDepth(MLP(PostNorm(X)))
    X ← ReversePartition(X)

    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 32,
        attention_channels: int = 64,
        attn_grid_window_size: Tuple[int, int] = (8, 8),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_norm_layer: Type[nn.Module] = nn.LayerNorm,
        post_norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_mlp: bool = False,
        use_normalised_qk: bool = True,
    ) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_heads : int, optional
             Number of attention heads, by default 32
        attention_channels : int
            Number of channels used for attention computations.
            It should be divisible by num_heads, by default 64
        attn_grid_window_size : Tuple[int, int], optional
             Grid/Window size to be utilized, by default (8, 8)
        attn_drop : float, optional
             Dropout ratio of attention weight, by default 0.0
        proj_drop : float, optional
            Dropout ratio of output, by default 0.0
        drop_path : float, optional
            Stochastic depth, by default 0.0
        pre_norm_layer : Type[nn.Module], optional
            Pre norm layer, by default nn.LayerNorm
        post_norm_layer : Type[nn.Module], optional
            Post norm layer, by default nn.LayerNorm
        use_mlp : bool, optional
            MLP to be used after the attention, by default False
        use_normalised_qk : bool, optional
            Normalise queries and keys as done in Metnet 3, by default True.

        Notes:
        -----
        Specific to Metnet 3 implementation
        """
        super().__init__()
        # Save parameters
        self.attn_grid_window_size: Tuple[int, int] = attn_grid_window_size

        rel_attn_bias = RelativePositionBias(attn_size=attn_grid_window_size, num_heads=num_heads)
        # Init layers
        self.attention = MultiheadSelfAttention2D(
            in_channels=in_channels,
            attention_channels=attention_channels,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_normalised_qk=use_normalised_qk,
            rel_attn_bias=rel_attn_bias,
        )
        self.pre_norm_layer = pre_norm_layer(attn_grid_window_size)  # Norm along windows
        self.post_norm_layer = post_norm_layer(attn_grid_window_size)

        if use_mlp:
            self.mlp = PointwiseMLP(in_channels=in_channels)
        else:
            self.mlp = nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def partition_function(self, X: torch.Tensor):
        """
        Partition function.

        To be overridden by block or grid partition

        Parameters
        ----------
        X : torch.Tensor
            Input tensor

        Raises:
        ------
        NotImplementedError
            Should not be called without implementation in the child class
        """
        raise NotImplementedError

    def reverse_function(
        self,
        partitioned_input: torch.Tensor,
        original_size: Tuple[int, int],
    ):
        """
        Undo Partition

        To be overridden by functions reversing the block or grid partitions

        Parameters
        ----------
        partitioned_input : torch.Tensor
            Partitioned input
        original_size : Tuple[int, int]
            Original Input size

        Raises:
        ------
        NotImplementedError
            Should not be called without implementation in the child class
        """
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of the shape [N, C, H, W].

        Returns:
        -------
        torch.Tensor
            Output tensor of the shape [N, C, H, W].
        """
        # Save original shape
        _, C, H, W = X.shape
        # Perform partition
        input_partitioned = self.partition_function(X)

        # Perform normalization, attention, and dropout
        output = input_partitioned + self.drop_path(
            self.attention(self.pre_norm_layer(input_partitioned))
        )

        # Perform normalization, MLP, and dropout
        output = output + self.drop_path(self.mlp(self.post_norm_layer(output)))

        # Reverse partition
        output = self.reverse_function(output, (H, W))
        return output


class BlockAttention(PartitionAttention):
    """
    Block Attention.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 32,
        attention_channels: int = 64,
        attn_grid_window_size: Tuple[int, int] = (8, 8),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_norm_layer: Type[nn.Module] = nn.LayerNorm,
        post_norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_mlp: bool = False,
        use_normalised_qk: bool = True,
    ) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_heads : int, optional
             Number of attention heads, by default 32
        attention_channels : int
            Number of channels used for attention computations.
            It should be divisible by num_heads, by default 64
        attn_grid_window_size : Tuple[int, int], optional
             Grid/Window size for attention, by default (8, 8)
        attn_drop : float, optional
             Dropout ratio of attention weight, by default 0.0
        proj_drop : float, optional
            Dropout ratio of output, by default 0.0
        drop_path : float, optional
            Stochastic depth, by default 0.0
        pre_norm_layer : Type[nn.Module], optional
            Pre norm layer, by default nn.LayerNorm
        post_norm_layer : Type[nn.Module], optional
            Post norm layer, by default nn.LayerNorm
        use_mlp : bool, optional
            MLP to be used after the attention, by default False
        use_normalised_qk : bool, optional
            Normalise queries and keys as done in Metnet 3, by default True.
        """
        super().__init__(
            in_channels,
            num_heads,
            attention_channels,
            attn_grid_window_size,
            attn_drop,
            proj_drop,
            drop_path,
            pre_norm_layer,
            post_norm_layer,
            use_mlp,
            use_normalised_qk,
        )

    def partition_function(self, input: Tensor) -> torch.Tensor:
        """
        Block partition function.

        Parameters
        ----------
        input : Tensor
            input (torch.Tensor): Input tensor of the shape [N, C, H, W].

        Returns:
        -------
        torch.Tensor
            blocks (torch.Tensor): Unfolded input tensor of the shape
            [N * blocks, partition_size[0], partition_size[1], C].
        """
        # Get size of input
        N, C, H, W = input.shape
        # Unfold input
        blocks = input.view(
            N,
            C,
            H // self.attn_grid_window_size[0],
            self.attn_grid_window_size[0],
            W // self.attn_grid_window_size[1],
            self.attn_grid_window_size[1],
        )
        # Permute and reshape to
        # [N * blocks, channels, self.attn_grid_window_size[0], self.attn_grid_window_size[1]]
        blocks = (
            blocks.permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(-1, C, self.attn_grid_window_size[0], self.attn_grid_window_size[1])
        )
        return blocks

    def reverse_function(
        self,
        partitioned_input: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Reverses the block partition.

        Parameters
        ----------
        partitioned_input : torch.Tensor
            Block tensor of the shape
            [N * partitioned_input, C, partition_size[0], partition_size[1]].
        original_size : Tuple[int, int]
            Original shape.

        Returns:
        -------
        torch.Tensor
            output (torch.Tensor): Folded output tensor of the shape
            [N, C, original_size[0], original_size[1]].
        """
        # Get height and width
        H, W = original_size
        # Compute original batch size
        N = int(
            partitioned_input.shape[0]
            / (H * W / self.attn_grid_window_size[0] / self.attn_grid_window_size[1])
        )
        # Fold grid tensor
        output = partitioned_input.view(  # TODO: Verify
            N,
            -1,
            H // self.attn_grid_window_size[0],
            W // self.attn_grid_window_size[1],
            self.attn_grid_window_size[0],
            self.attn_grid_window_size[1],
        )
        output = output.contiguous().view(N, -1, H, W)
        return output


class GridAttention(PartitionAttention):
    """
    Grid Attention
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 32,
        attention_channels: int = 64,
        attn_grid_window_size: Tuple[int, int] = (8, 8),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_norm_layer: Type[nn.Module] = nn.LayerNorm,
        post_norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_mlp: bool = False,
        use_normalised_qk: bool = True,
    ) -> None:
        """
        Constructor Method

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_heads : int, optional
             Number of attention heads, by default 32
        attention_channels : int
            Number of channels used for attention computations.
            It should be divisible by num_heads, by default 64
        attn_grid_window_size : Tuple[int, int], optional
             Grid/Window size to be utilized, by default (8, 8)
        attn_drop : float, optional
             Dropout ratio of attention weight, by default 0.0
        proj_drop : float, optional
            Dropout ratio of output, by default 0.0
        drop_path : float, optional
            Stochastic depth, by default 0.0
        pre_norm_layer : Type[nn.Module], optional
            Pre norm layer, by default nn.LayerNorm
        post_norm_layer : Type[nn.Module], optional
            Post norm layer, by default nn.LayerNorm
        use_mlp : bool, optional
            MLP to be used after the attention, by default False
        use_normalised_qk : bool, optional
            Normalise queries and keys as done in Metnet 3, by default True.
        """
        super().__init__(
            in_channels,
            num_heads,
            attention_channels,
            attn_grid_window_size,
            attn_drop,
            proj_drop,
            drop_path,
            pre_norm_layer,
            post_norm_layer,
            use_mlp,
            use_normalised_qk,
        )

    def partition_function(self, input: Tensor) -> torch.Tensor:
        """
        Grid partition function.

        Parameters
        ----------
        input : Tensor
            Input tensor of the shape [N, C, H, W].

        Returns:
        -------
        torch.Tensor
            Unfolded input tensor of the shape
            [N * grids, grid_size[0], grid_size[1], C].
        """
        # Get size of input
        N, C, H, W = input.shape
        # Unfold input
        grid = input.view(
            N,
            C,
            self.attn_grid_window_size[0],
            H // self.attn_grid_window_size[0],
            self.attn_grid_window_size[1],
            W // self.attn_grid_window_size[1],
        )
        # Permute and reshape [N * (H // self.attn_grid_window_size[0]) * (W // self.attn_grid_window_size[1]), C, self.attn_grid_window_size[0], self.attn_grid_window_size[1]]  # noqa
        grid = (
            grid.permute(0, 3, 5, 1, 2, 4)
            .contiguous()
            .view(-1, C, self.attn_grid_window_size[0], self.attn_grid_window_size[1])
        )
        return grid

    def reverse_function(
        self,
        partitioned_input: torch.Tensor,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Reverses the grid partition.

        Parameters
        ----------
        partitioned_input : torch.Tensor
            Grid tensor of the shape
            [N * partitioned_input, C, partition_size[0], partition_size[1]].
        original_size : Tuple[int, int]
            Original shape.

        Returns:
        -------
        torch.Tensor
            Folded output tensor of the shape [N, C, original_size[0], original_size[1]].
        """
        # Get height, width, and channels
        (H, W), C = original_size, partitioned_input.shape[1]
        # Compute original batch size
        N = int(
            partitioned_input.shape[0]
            / (H * W / self.attn_grid_window_size[0] / self.attn_grid_window_size[1])
        )
        # Fold partitioned_input tensor
        output = partitioned_input.view(  # TODO: Verify
            N,
            C,
            H // self.attn_grid_window_size[0],
            W // self.attn_grid_window_size[1],
            self.attn_grid_window_size[0],
            self.attn_grid_window_size[1],
        )
        output = output.permute(0, 1, 3, 5, 4, 2).contiguous().view(N, C, H, W)
        return output
