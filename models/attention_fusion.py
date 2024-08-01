import torch
from torch import nn

class LocalFusion(nn.Module):
    def __init__(self, att_in_dim=3, num_categories=6, max_pool_ksize1=4, max_pool_ksize2=2, encoder_dims=[8, 16]):
        super().__init__()
        self.num_categories = num_categories
        self.att_in_dim = att_in_dim

        self.attention_fusion = nn.ModuleList([Self_Attn(in_dim=att_in_dim, max_pool_ksize1=max_pool_ksize1, max_pool_ksize2=max_pool_ksize2, encoder_dims=encoder_dims) for _ in range(num_categories)])

    def forward(self, x, color_naming_probs=None, q=None):

        # Using the average to compute the blending
        if color_naming_probs is None:
            # Using the same input tensor for query, key, and value
            if q is None:
                return torch.mean(torch.stack([att(x_color, q=x) for att, x_color in zip(self.attention_fusion, x)], dim=0))
            else:
                return torch.mean(torch.stack([att(x_color, q=q) for att, x_color in zip(self.attention_fusion, x)], dim=0))

        # Using the color naming probabilities to compute the blending. Weighted average with color naming probs as
        # weights.
        else:
            color_naming_probs = (color_naming_probs > 0.20).float()
            color_naming_avg = torch.sum(color_naming_probs, dim=0).unsqueeze(1).repeat(1, 3, 1, 1)
            color_naming_probs = color_naming_probs.unsqueeze(2).repeat(1, 1, 3, 1, 1)

            # Using the same input tensor for query, key, and value
            if q is None:
                out = torch.stack([att(x_color, q=x) for att, x_color in zip(self.attention_fusion, x)], dim=0)
            else:
                out = torch.stack([att(x_color, q=q) for att, x_color in zip(self.attention_fusion, x)], dim=0)

            out = torch.sum(out * color_naming_probs, dim=0) / color_naming_avg
            return torch.clip(out, 0, 1)

class Self_Attn(nn.Module):
    def __init__(self, in_dim, max_pool_ksize1=4, max_pool_ksize2=2, encoder_dims=[8, 16]):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.max_pool_ksize1 = max_pool_ksize1
        self.max_pool_ksize2 = max_pool_ksize2
        self.down_ratio = max_pool_ksize1 * max_pool_ksize2

        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=encoder_dims[0], kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(in_channels=encoder_dims[0], out_channels=encoder_dims[1], kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=encoder_dims[0], kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(in_channels=encoder_dims[0], out_channels=encoder_dims[1], kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=encoder_dims[0], kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(in_channels=encoder_dims[0], out_channels=encoder_dims[1], kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=encoder_dims[1], out_channels=encoder_dims[0], kernel_size=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(in_channels=encoder_dims[0], out_channels=encoder_dims[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=encoder_dims[0], out_channels=3, kernel_size=1),
            nn.ReLU())

        self.last_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.max_pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q=None):

        if q is None:
            q = x

        m_batch_size, C, width, height = x.size()
        proj_query = self.query_conv(q).view(m_batch_size, -1, int((width//self.down_ratio)*(height//self.down_ratio))).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batch_size, -1, int((width//self.down_ratio)*(height//self.down_ratio)))
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batch_size, -1, int((width//self.down_ratio)*(height//self.down_ratio)))

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batch_size, 16, int(width//self.down_ratio), int(height//self.down_ratio))

        out = self.upsample(out)
        upsampled_layer = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=False)
        out = upsampled_layer(out)

        out = self.last_conv(out)

        out = out + x
        return out