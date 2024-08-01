from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

class BackboneL2SSIMLoss(nn.Module):
    def __init__(self, ssim_window_size=5, alpha=0.5):
        super(BackboneL2SSIMLoss, self).__init__()
        self.ssim_window_size = ssim_window_size
        self.alpha = alpha

        self.ssim_loss = SSIM(kernel_size=ssim_window_size).cuda()
        self.l2_loss = nn.MSELoss()

    def forward(self, backbone, prediction, target):
        ssim_loss_pred = (1.0 - self.ssim_loss(prediction, target))
        l2_loss_pred = self.l2_loss(prediction, target)
        l2_loss_backbone = self.l2_loss(backbone, target)

        return self.alpha*l2_loss_backbone + l2_loss_pred + ssim_loss_pred

def get_criterion(criterion_config):
    criterion_type = criterion_config.type
    if criterion_type == 'backbone-L2-SSIM':
        return BackboneL2SSIMLoss(**criterion_config['params'])
    #TODO: Add more criterion types here (L1, L2 ...)
    else:
        raise ValueError(f"Unsupported criterion type: {criterion_type}")