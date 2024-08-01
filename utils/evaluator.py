import torch
import logging
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from lpips import LPIPS
from utils.deltaE import deltaEab, deltaE00

class Evaluator():
    def __init__(self, dataloader, metrics, split_name, log_dirpath, best_metric):
        self.dataloader = dataloader
        self._create_metrics(metrics)
        self.split_name = split_name
        self.log_dirpath = log_dirpath
        self.best_metric = best_metric
        self.best_value = 0

    def _create_metrics(self, metrics):
        self.metrics = {}
        self.cumulative_values = {}
        for metric in metrics:
            if metric.type == 'PSNR':
                self.metrics['PSNR'] = PSNR(**metric.params).cuda()
                self.cumulative_values['PSNR'] = 0
            elif metric.type == 'SSIM':
                self.metrics['SSIM'] = SSIM(**metric.params).cuda()
                self.cumulative_values['SSIM'] = 0
            elif metric.type == 'LPIPS':
                self.metrics['LPIPS'] = LPIPS(**metric.params).cuda()
                self.cumulative_values['LPIPS'] = 0
            elif metric.type == 'deltaEab':
                self.metrics['deltaEab'] = deltaEab()
                self.cumulative_values['deltaEab'] = 0
            elif metric.type == 'deltaE00':
                self.metrics['deltaE00'] = deltaE00()
                self.cumulative_values['deltaE00'] = 0
            else:
                raise NotImplementedError(f"Metric {metric.type} not implemented")

    def _compute_metrics(self, input_image, target_image):
        for name, metric in self.metrics.items():
            self.cumulative_values[name] += metric(input_image, target_image)

    def _compute_average_metrics(self):
        avg_metrics = {}
        for name, value in self.cumulative_values.items():
            avg_metrics[name] = float(value / len(self.dataloader))
        return avg_metrics

    def _reset_metrics(self):
        for metric in self.metrics:
            self.cumulative_values[metric] = 0

    def __call__(self, model, save_results=True):
        model.eval()
        self._reset_metrics()
        with torch.no_grad():
            for data in self.dataloader:
                input_image, target_image, name = data['input_image'], data['target_image'], data['name']

                self._compute_metrics(input_image.cuda(), target_image.cuda())

        avg_metrics = self._compute_average_metrics()
        logging.info(f"{self.split_name} metrics: " + ", ".join([f'{key}: {value:.4f}' for key, value in avg_metrics.items()]))

        if (avg_metrics[self.best_metric] > self.best_value) and save_results:
            self.best_value = avg_metrics[self.best_metric]
            torch.save({**{'model_state_dict': model.state_dict()}, **avg_metrics},
                       f"{self.log_dirpath}/{self.split_name}_best_model.pth")
            logging.info(f"New best model saved at {self.log_dirpath}/{self.split_name}_best_model.pth")
