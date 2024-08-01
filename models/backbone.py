"""
backbone.py - Contains the backbone of the model.
(It is based on LPIENet and CURL's backbone)

Perceptual Image Enhancement for Smartphone Real-Time Applications
https://github.com/mv-lab/AISP

CURL: Neural Curve Layers for Global Image Enhancement
https://github.com/sjmoran/CURL

David Serrano (dserrano@cvc.uab.cat)
May 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super(AttentionBlock, self).__init__()
        self._spatial_attention_conv = nn.Conv2d(2, dim, kernel_size=3, padding=1)

        # Channel attention MLP
        self._channel_attention_conv0 = nn.Conv2d(1, dim, kernel_size=1, padding=0)
        self._channel_attention_conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)

        self._out_conv = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {x.shape}.")

        # Spatial attention
        mean = torch.mean(x, dim=1, keepdim=True)  # Mean/Max on C axis
        max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([mean, max], dim=1)  # [B, 2, H, W]
        spatial_attention = self._spatial_attention_conv(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention) * x

        # NOTE: This differs from CBAM as it uses Channel pooling, not spatial pooling!
        # In a way, this is 2x spatial attention
        channel_attention = torch.relu(self._channel_attention_conv0(mean))
        channel_attention = self._channel_attention_conv1(channel_attention)
        channel_attention = torch.sigmoid(channel_attention) * x

        attention = torch.cat([spatial_attention, channel_attention], dim=1)  # [B, 2*dim, H, W]
        attention = self._out_conv(attention)
        return x + attention


class InverseBlock(nn.Module):
    def __init__(self, input_channels: int, channels: int):
        super(InverseBlock, self).__init__()

        self._conv0 = nn.Conv2d(input_channels, channels, kernel_size=1)
        self._dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self._conv2 = nn.Conv2d(input_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self._conv0(x)
        features = F.elu(self._dw_conv(features))
        features = self._conv1(features)

        x = torch.relu(self._conv2(x))
        return x + features


class BaseBlock(nn.Module):
    def __init__(self, channels: int):
        super(BaseBlock, self).__init__()

        self._conv0 = nn.Conv2d(channels, channels, kernel_size=1)
        self._dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=1)

        self._conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self._conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self._conv0(x)
        features = F.elu(self._dw_conv(features))
        features = self._conv1(features)
        x = x + features

        features = F.elu(self._conv2(x))
        features = self._conv3(features)
        return x + features


class AttentionTail(nn.Module):
    def __init__(self, channels: int):
        super(AttentionTail, self).__init__()

        self._conv0 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self._conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        attention = torch.relu(self._conv0(x))
        attention = torch.relu(self._conv1(attention))
        attention = torch.sigmoid(self._conv2(attention))
        return x * attention

class Flatten(nn.Module):

    def forward(self, x):
        """Flatten a Tensor to a Vector

        :param x: Tensor
        :returns: 1D Tensor
        :rtype: Tensor

        """
        return x.view(x.size()[0], -1)

class ResidualConnection(nn.Module):
    def __init__(self, in_channels):
        super(ResidualConnection, self).__init__()

        self.in_channels = in_channels

        self.midnet2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 2, 2),
            nn.LeakyReLU()
        )

        self.midnet4 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 4, 4),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 4, 4),
            nn.LeakyReLU()
        )

        self.globnet = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 64)
        )

        self.conv_fuse = nn.Conv2d(in_channels=192+in_channels, out_channels=in_channels, kernel_size=1)
    def forward(self, x):

        x_midnet2 = self.midnet2(x)
        x_midnet4 = self.midnet4(x)
        x_global = self.globnet(x).unsqueeze(2).unsqueeze(3)
        x_global = x_global.repeat(1, 1, x_midnet2.shape[2], x_midnet2.shape[3])

        x_fuse = torch.cat((x, x_midnet2, x_midnet4, x_global), dim=1)
        x_out = self.conv_fuse(x_fuse)

        return x_out

class Backbone(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, encoder_dims: List[int], decoder_dims: List[int]):
        super(Backbone, self).__init__()

        if len(encoder_dims) != len(decoder_dims) + 1 or len(decoder_dims) < 1:
            raise ValueError(f"Unexpected encoder and decoder dims: {encoder_dims}, {decoder_dims}.")

        if input_channels != output_channels:
            raise NotImplementedError()

        encoders = []
        for i, encoder_dim in enumerate(encoder_dims):
            input_dim = input_channels if i == 0 else encoder_dims[i - 1]
            encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, encoder_dim, kernel_size=3, padding=1),
                    BaseBlock(encoder_dim),
                    BaseBlock(encoder_dim),
                    AttentionBlock(encoder_dim),
                )
            )
        self._encoders = nn.ModuleList(encoders)

        decoders = []
        for i, decoder_dim in enumerate(decoder_dims):
            input_dim = encoder_dims[-1] if i == 0 else decoder_dims[i - 1] + encoder_dims[-i - 1]
            decoders.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, decoder_dim, kernel_size=3, padding=1),
                    BaseBlock(decoder_dim),
                    BaseBlock(decoder_dim),
                    AttentionBlock(decoder_dim),
                )
            )
        self._decoders = nn.ModuleList(decoders)

        self._inverse_bock = InverseBlock(encoder_dims[0] + decoder_dims[-1], output_channels)
        self._attention_tail = AttentionTail(output_channels)

        residual_connections = []
        for i, decoder_dim in enumerate(encoder_dims):
            residual_connections.append(
                ResidualConnection(in_channels=decoder_dim)
            )
        self._residual_connections = nn.ModuleList(residual_connections)
    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {x.shape}.")
        global_residual = x

        encoder_outputs, residual_connections = [], []
        for i, encoder in enumerate(self._encoders):
            x = encoder(x)
            if i != len(self._encoders) - 1:
                encoder_outputs.append(x)
                residual_connections.append(self._residual_connections[i](x))
                x = F.max_pool2d(x, kernel_size=2)

        encoder_outputs.reverse()
        residual_connections.reverse()
        for i, decoder in enumerate(self._decoders):
            x = decoder(x)
            x = nn.Upsample(size=encoder_outputs[i].shape[2:], mode='bilinear', align_corners=False)(x)
            x = torch.cat([x, residual_connections[i]], dim=1)

        x = self._inverse_bock(x)
        x = self._attention_tail(x)
        return torch.clip(x + global_residual, 0, 1)