"""Condition with time how MetNet-22 does it, with FiLM layers"""

import einops
import torch
from torch import nn as nn


class ConditionWithTimeMetNet2(nn.Module):
    """Compute Scale and bias for conditioning on time"""

    def __init__(self, forecast_steps: int, hidden_dim: int, num_feature_maps: int):
        """
        Compute the scale and bias factors for conditioning convolutional blocks on the forecast time

        Args:
            forecast_steps: Number of forecast steps
            hidden_dim: Hidden dimension size
            num_feature_maps: Max number of channels in the blocks, to generate enough scale+bias values
                This means extra values will be generated, but keeps implementation simpler
        """
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_feature_maps = num_feature_maps
        self.lead_time_network = nn.ModuleList(
            [
                nn.Linear(in_features=forecast_steps, out_features=hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=2 * num_feature_maps),
            ]
        )

    def forward(self, x: torch.Tensor, timestep: int) -> [torch.Tensor, torch.Tensor]:
        """
        Get the scale and bias for the conditioning layers

        From the FiLM paper, each feature map (i.e. channel) has its own scale and bias layer, so needs
        a scale and bias for each feature map to be generated

        Args:
            x: The Tensor that is used
            timestep: Index of the timestep to use, between 0 and forecast_steps

        Returns:
            2 Tensors of shape (Batch, num_feature_maps)
        """
        # One hot encode the timestep
        timesteps = torch.zeros(x.size()[0], self.forecast_steps, dtype=torch.long).type_as(x)
        timesteps[:, timestep] = 1
        # Get scales and biases
        for layer in self.lead_time_network:
            timesteps = layer(timesteps)
        scales_and_biases = timesteps
        scales_and_biases = einops.rearrange(
            scales_and_biases, "b (block sb) -> b block sb", block=self.num_feature_maps, sb=2
        )
        return scales_and_biases[:, :, 0], scales_and_biases[:, :, 1]
