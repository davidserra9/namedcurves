from models.attention_fusion import LocalFusion
from models.bezier_control_point_estimator import BCPE
from models.color_naming import ColorNaming
from models.backbone import Backbone
from torch import nn

from PIL import Image
from torchvision.transforms import functional as TF
import torch

class NamedCurves(nn.Module):
    def __init__(self, configs: dict, device='cuda'):
        super().__init__()
        self.model_configs = configs

        self.backbone = Backbone(**configs['backbone']['params'])
        self.color_naming = ColorNaming(num_categories=configs['color_naming']['num_categories'], device=device)
        self.bcpe = BCPE(**configs['bezier_control_points_estimator']['params'])
        self.local_fusion = LocalFusion(**configs['local_fusion']['params'])

    def forward(self, x, return_backbone=False):
        x_backbone = self.backbone(x)
        cn_probs = self.color_naming(x_backbone)
        x_global = self.bcpe(x_backbone, cn_probs)
        out = self.local_fusion(x_global, cn_probs, q=x_backbone)
        if return_backbone:
            return out, x_backbone
        return out