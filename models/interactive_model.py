"""NamedCurves model with interactive functionality. This version builds upon model.py and bezier_control_point_estimator.py by incorporating additional parameters."""

from models.attention_fusion import LocalFusion
from models.color_naming import ColorNaming
from models.backbone import Backbone
from torch import nn

from PIL import Image
from torchvision.transforms import functional as TF
import torch

class NamedCurves(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.model_configs = configs

        self.backbone = Backbone(**configs['backbone']['params'])
        self.color_naming = ColorNaming(num_categories=configs['color_naming']['num_categories'])
        self.bcpe = BCPE(**configs['bezier_control_points_estimator']['params'])
        self.local_fusion = LocalFusion(**configs['local_fusion']['params'])

    def forward(self, x, return_backbone=False, return_curves=False, control_points=None):
        x_backbone = self.backbone(x)
        cn_probs = self.color_naming(x_backbone)

        if return_curves:
            x_global, control_points = self.bcpe(x_backbone, cn_probs, return_control_points=return_curves, control_points=control_points)
        else:
            x_global = self.bcpe(x_backbone, cn_probs, control_points=control_points)

        out = self.local_fusion(x_global, cn_probs, q=x_backbone)

        if return_backbone:
            return out, x_backbone
        if return_curves:
            return out, control_points
        return out

class ContextualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU())

    def forward(self, x):
        return self.main(x)

class BezierColorBranch(nn.Module):
    def __init__(self, num_control_points=10):
        super().__init__()
        self.num_control_points = num_control_points # +1, (0, 0) point
        self.color_branch = nn.Sequential(
            nn.Conv2d(65, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3 * self.num_control_points, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.sigmoid = nn.Sigmoid()

    def create_control_points(self, x):
        x = torch.cumsum(torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1), dim=-1)
        x = torch.stack([x, torch.linspace(0, 1, steps=self.num_control_points+1).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1).cuda()], dim=-1)
        return x

    def forward(self, x):
        x = self.color_branch(x).view(x.size(0), 3, self.num_control_points)
        x = self.sigmoid(x)
        x = x / torch.sum(x, dim=2)[..., None]
        x = self.create_control_points(x)
        return x

class BCPE(nn.Module):
    def __init__(self, num_categories=6, num_control_points=10):
        super().__init__()

        self.contextual_feature_extractor = ContextualFeatureExtractor()
        self.color_branches = nn.ModuleList([BezierColorBranch(num_control_points) for _ in range(num_categories)])

    def binomial_coefficient(self, n, k):
        """
        Calculate the binomial coefficient (n choose k).
        """
        if k < 0 or k > n:
            return 0.0
        result = 1.0
        for i in range(min(k, n - k)):
            result *= (n - i)
            result //= (i + 1)
        return result

    def apply_cubic_bezier(self, x, control_points):

        n = control_points.shape[2]
        output = torch.zeros_like(x)
        for j in range(n):
            output += control_points[..., j, 0].view(control_points.shape[0], control_points.shape[1], 1, 1) * self.binomial_coefficient(n - 1, j) * (1 - x) ** (n - 1 - j) * x ** j
        return output

    def forward(self, x, cn_probs, return_control_points=False, control_points=None):
        feat = self.contextual_feature_extractor(x)
        bezier_control_points = [color_branch(torch.cat((feat, color_probs.unsqueeze(1)), dim=1).float()) for color_branch, color_probs in zip(self.color_branches, cn_probs)]
        
        if control_points is not None:
            bezier_control_points = control_points

        global_adjusted_images = torch.stack([self.apply_cubic_bezier(x, control_points) for control_points in bezier_control_points], dim=0)
        
        if return_control_points:
            return global_adjusted_images, bezier_control_points
        
        return global_adjusted_images