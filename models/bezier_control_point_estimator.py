"""
bezier_control_point_estimator.py - Contains the Bezier Control Point Estimator model.
The Bezier Control Point Estimator estimates the set of control points that define the Bezier curve for each color name.

David Serrano (dserrano@cvc.uab.cat)
May 2024
"""

import torch
from torch import nn

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

    def forward(self, x, cn_probs):
        feat = self.contextual_feature_extractor(x)
        bezier_control_points = [color_branch(torch.cat((feat, color_probs.unsqueeze(1)), dim=1).float()) for color_branch, color_probs in zip(self.color_branches, cn_probs)]
        global_adjusted_images = torch.stack([self.apply_cubic_bezier(x, control_points) for control_points in bezier_control_points], dim=0)
        return global_adjusted_images
