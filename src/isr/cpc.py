"""
Contrastive Prediction Coding model for feature extraction.
code by @Pstva
"""

import torch
from torch import nn
from torch.nn import functional as F
from math import ceil, floor


def make_conv1d_layer(c_in, c_out, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
    )


class ResnetBlock(nn.Module):
    """
    resnet block for Encoder
    """
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 kernel_size: int,
                 stride: int,
                 extra_conv_n: int = 1
                 ):
        """
        :param c_in: input channels
        :param c_out: output channels
        :param kernel_size: kernel size of the convolution
        :param stride: stride of the convolution
        :param extra_conv_n: number of convolutional layers added after basic
            conv layer that downsample signal

        Padding is also added for reducing the signal dimensionality only by
        stride. Padding is calculated as follows:
            padding = (stride * (output_dim - 1) - input_dim + filter_size) / 2
        We want:
            output_dim = input_dim // stride,
        so:
            padding = (filter_size - stride) / 2
        """
        super().__init__()
        pad_left = ceil((kernel_size - stride) / 2)
        pad_right = floor((kernel_size - stride) / 2)

        # skip connection (downsample input) if needed
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.ConstantPad1d((pad_left, pad_right), 0),
                nn.Conv1d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.downsample = None

        # basic block
        pad_left, pad_right = ceil((kernel_size - stride) / 2), floor(
            (kernel_size - stride) / 2
        )
        block = [
            # тут такой паддинг из-за четных сверток
            nn.ConstantPad1d((pad_left, pad_right), 0),
            nn.Conv1d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(c_out),
            nn.ReLU()
        ]
        for _ in range(extra_conv_n):
            block.extend(make_conv1d_layer(c_in=c_out, c_out=c_out,
                                           kernel_size=3, stride=1, padding=1))

        # delete last nn.ReLU, as it will be added in forward
        self.block = nn.Sequential(*block[:-1])

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: torch.tensor of shape [B, C_in, N]
        where
            B - batch size
            C - channels
            N - length
        :return: torch.tensor of shape [B, C_out, N // stride]
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.block(x)
        out += identity
        return F.relu(out)


class CPCResnetEncoder(nn.Module):
    """
    Encoder network for encoding the signal.
    Encoder consists of five convolutional layers with strides [5, 4, 2, 2, 2],
    filter-sizes [10, 8, 4, 4, 4] and 512 hidden units with ReLU activations.
    Total downsampling factor of the network is 160, so, for the signal with
    sample_rate=16000, there is a feature vector for every 10ms of speech.
    """
    def __init__(self,
                 hidden_dim: int = 512,
                 extra_conv_n: int = 1):
        """
        :param hidden_dim: the dimensionality of the output embeddings
        :param extra_conv_n: extra convolutional layers added inside each
            residual block (added after the main conv.layer that downsamples
            audio)
        """
        super().__init__()
        self.model = nn.Sequential(
            ResnetBlock(1, hidden_dim // 16, 10, 5, extra_conv_n),
            ResnetBlock(hidden_dim // 16, hidden_dim // 8, 8, 4, extra_conv_n),
            ResnetBlock(hidden_dim // 8, hidden_dim // 4, 4, 2, extra_conv_n),
            ResnetBlock(hidden_dim // 4, hidden_dim // 2, 4, 2, extra_conv_n),
            ResnetBlock(hidden_dim // 2, hidden_dim, 4, 2, extra_conv_n)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: torch.tensor of shape [B, 1, N],
        where
            B - batch size
            N - the signal length
        (second dimension stands for the number of channels, which is 1 for
        our speech data)
        :return: torch.tensor of shape [B, hidden_dim, N // 160]
        """
        x = self.model(x)
        return x


class ConvLayerNormBlock(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 kernel_size: int,
                 stride: int,
                 padding: int
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.layer_norm = nn.LayerNorm(c_out)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = self.layer_norm(x)
        x = x.transpose(-1, -2)
        x = self.activation(x)
        return x


class CPCEncoderLayerNorm(nn.Module):
    """
    CPCResnetEncoderBig with waveform normalization and Layernorm like in
    wav2vec2.0
    """
    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        self.model = nn.Sequential(
            ConvLayerNormBlock(1, hidden_dim, 10, 5, 3),
            ConvLayerNormBlock(hidden_dim, hidden_dim, 8, 4, 3),
            ConvLayerNormBlock(hidden_dim, hidden_dim, 4, 2, 1),
            ConvLayerNormBlock(hidden_dim, hidden_dim, 4, 2, 1),
            ConvLayerNormBlock(hidden_dim, hidden_dim, 4, 2, 1)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: torch.tensor of shape [B, 1, N],
        where
            B - batch size
            N - the signal length
        (second dimension stands for the number of channels, which is 1 for
        our speech data)
        :return: torch.tensor of shape [B, hidden_dim, N // 160]
        """
        # waveform normalization
        m = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        x_norm = (x - m) / torch.sqrt(var + 1e-7)
        # extracting encoder vectors
        return self.model(x_norm)


class CPC(nn.Module):
    """
    full model for Contrastive Predicting Coding
    """
    def __init__(
        self,
        emb_dim: int = 512,
        context_dim: int = 256,
        encoder: str = "LayerNormEncoder",
        rnn: str = "GRU"
    ):
        """
        :param emb_dim: size of embeddings (output dim after encoder)
        :param context_dim: size of context embeddings (output dim after rnn)
        :param encoder: type of Encoder in CPC model
        :param rnn: type of RNN in CPC model
        """
        super().__init__()
        encoders = {
            "ResnetEncoder": CPCResnetEncoder,
            "LayerNormEncoder": CPCEncoderLayerNorm
        }

        context_networks = {"GRU": nn.GRU, "LSTM": nn.LSTM}

        if encoder not in encoders:
            raise ValueError("Unknown Encoder, can be one of " +
                             f"{encoders.keys()}")
        if rnn not in context_networks:
            raise ValueError("Unknown RNN, can be one of" +
                             f"{context_networks.keys()}")

        self.encoder = encoders[encoder](emb_dim)
        self.rnn = context_networks[rnn](
            input_size=emb_dim, hidden_size=context_dim, batch_first=True)

    def forward(self, x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        :param x: torch.tensor of shape [B, C, N]
        where
            B - batch size
            C - number of audio channels (=1)
            N - length of audio signal
        :return:
            Tuple[z_emb, z_emb]:
                z_emb: torch.tensor of shape [B, N // 160, emb_dim] -
                    output from Encoder
                z_pred: torch.tensor of shape [B, N // 160, context_dim] -
                    output from RNN
        """
        # embeddings for the signals
        # shape [B, N, emb_dim]
        z_emb = self.encoder(x).transpose(1, 2)
        # context embeddings from rnn
        # shape [B, N, context_dim]
        c_emb, _ = self.rnn(z_emb)
        return z_emb, c_emb


# usage examples
if __name__ == "__main__":
    sig = torch.randn(size=(1, 1, 20480))
    # first checkpoint
    model = CPC(emb_dim=512, context_dim=256, encoder="ResnetEncoder",
                rnn="GRU")
    model.load_state_dict(torch.load("cpc_base_aug_100h.pth"))
    z_emb, c_emb = model(sig)  # z_emb [1, 128, 512], c_emb[1, 128, 256]

    # first checkpoint
    model = CPC(emb_dim=512, context_dim=256, encoder="LayerNormEncoder",
                rnn="GRU")
    model.load_state_dict(torch.load("cpc_layernorm_aug_960h.pth"))
    z_emb, c_emb = model(sig)  # z_emb [1, 128, 512], c_emb[1, 128, 256]
