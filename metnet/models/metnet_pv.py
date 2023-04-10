import torch
import torch.nn as nn
import torch.nn.functional as F
from axial_attention import AxialAttention, AxialPositionalEmbedding
from huggingface_hub import PyTorchModelHubMixin

from metnet.layers import ConditionTime, ConvGRU, DownSampler, MetNetPreprocessor, TimeDistributed


class MetNetPV(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        image_encoder: str = "downsampler",
        input_channels: int = 12,
        sat_channels: int = 12,
        input_size: int = 256,
        output_channels: int = 12,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 1,
        num_att_layers: int = 1,
        forecast_steps: int = 48,
        temporal_dropout: float = 0.2,
        use_preprocessor: bool = True,
        num_pv_systems: int = 1000,
        pv_fc_out_channels: int = 256,
        pv_id_embedding_channels: int = 16,
        fc_1_channels: int = 256,
        **kwargs,
    ):
        super(MetNetPV, self).__init__()
        config = locals()
        config.pop("self")
        config.pop("__class__")
        self.config = kwargs.pop("config", config)
        sat_channels = self.config["sat_channels"]
        input_size = self.config["input_size"]
        input_channels = self.config["input_channels"]
        temporal_dropout = self.config["temporal_dropout"]
        image_encoder = self.config["image_encoder"]
        forecast_steps = self.config["forecast_steps"]
        hidden_dim = self.config["hidden_dim"]
        kernel_size = self.config["kernel_size"]
        num_layers = self.config["num_layers"]
        num_att_layers = self.config["num_att_layers"]
        output_channels = self.config["output_channels"]
        use_preprocessor = self.config["use_preprocessor"]
        num_pv_systems = self.config["num_pv_systems"]
        pv_fc_out_channels = self.config["pv_fc_out_channels"]
        pv_id_embedding_channels = self.config["pv_id_embedding_channels"]
        fc_1_channels = self.config["fc_1_channels"]

        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = output_channels

        if use_preprocessor:
            self.preprocessor = MetNetPreprocessor(
                sat_channels=sat_channels,
                crop_size=input_size,
                use_space2depth=True,
                split_input=True,
            )
            # Update number of input_channels with output from MetNetPreprocessor
            new_channels = sat_channels * 4  # Space2Depth
            new_channels *= 2  # Concatenate two of them together
            input_channels = input_channels - sat_channels + new_channels
        else:
            self.preprocessor = torch.nn.Identity()

        self.drop = nn.Dropout(temporal_dropout)
        if image_encoder in ["downsampler", "default"]:
            image_encoder = DownSampler(input_channels + forecast_steps)
        else:
            raise ValueError(f"Image_encoder {image_encoder} is not recognized")
        self.image_encoder = TimeDistributed(image_encoder)
        self.ct = ConditionTime(forecast_steps)
        self.temporal_enc = TemporalEncoder(
            image_encoder.output_channels, hidden_dim, ks=kernel_size, n_layers=num_layers
        )
        self.position_embedding = AxialPositionalEmbedding(
            dim=self.temporal_enc.out_channels, shape=(input_size // 4, input_size // 4)
        )
        self.temporal_agg = nn.Sequential(
            *[
                AxialAttention(dim=hidden_dim, dim_index=1, heads=8, num_dimensions=2)
                for _ in range(num_att_layers)
            ]
        )
        self.fc1 = nn.LazyLinear(out_features=fc_1_channels)
        self.head = nn.Linear(fc_1_channels, output_channels)
        # PV Auxiliary Input
        self.pv_fc1 = nn.Linear(num_pv_systems, out_features=pv_fc_out_channels)
        self.pv_system_id_embedding = nn.Embedding(
            num_embeddings=940, embedding_dim=pv_id_embedding_channels
        )

    def encode_timestep(self, x, pv_yield_history, fstep=1):
        # Preprocess Tensor
        x = self.preprocessor(x)

        # Condition Time
        x = self.ct(x, fstep)

        ##CNN
        x = self.image_encoder(x)

        # Temporal Encoder
        _, state = self.temporal_enc(self.drop(x))
        return self.temporal_agg(self.position_embedding(state))

    def forward(
        self, imgs: torch.Tensor, pv_yield_history, pv_system_id, lead_time: int = 0
    ) -> torch.Tensor:
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        """
        x_i = self.encode_timestep(imgs, pv_yield_history, lead_time)
        ## Reshape so can concat
        x_i = x_i.reshape(imgs.shape[0], -1)
        ## Add in PV Data
        pv_yield_history = F.relu(self.pv_fc1(pv_yield_history))
        pv_yield_history = pv_yield_history.reshape(imgs.shape[0], -1)
        # Add embedding
        embedding = self.pv_system_id_embedding(pv_system_id)
        embedding = embedding.reshape(imgs.shape[0], -1)
        x_i = torch.cat((x_i, pv_yield_history, embedding), dim=1)
        x_i = F.relu(self.fc1(x_i))
        res = self.head(x_i)
        return res


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1):
        super().__init__()
        self.out_channels = out_channels
        self.rnn = ConvGRU(in_channels, out_channels, (ks, ks), n_layers, batch_first=True)

    def forward(self, x):
        x, h = self.rnn(x)
        return (x, h[-1])


def feat2image(x, target_size=(128, 128)):
    "This idea comes from MetNet"
    x = x.transpose(1, 2)
    return x.unsqueeze(-1).unsqueeze(-1) * x.new_ones(1, 1, 1, *target_size)
