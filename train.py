import torch
import datetime
import os
import logging

import omegaconf

from utils.logger import prepare_logging
from torch.utils.data import DataLoader
from data.datasets import get_datasets
from models.model import NamingEnhancementModel
from utils.setup_optim_scheduler import get_optimizer_scheduler
from utils.evaluator import Evaluator
from utils.setup_criterion import get_criterion
from utils.trainer import Trainer


def main(config: omegaconf.DictConfig):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.train.cuda_visible_device)
    save_path = prepare_logging()

    logging.info(f"Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Saving logs to {save_path}")
    logging.info(f"Config file: {OmegaConf.to_yaml(config)}")

    train_dataset, valid_dataset, test_dataset = get_datasets(config.data)

    msg = f"Training with {len(train_dataset)} image pairs"
    if valid_dataset is not None:
        msg += f", validating with {len(valid_dataset)} image pairs"
    msg += f" and testing with {len(test_dataset)} image pairs."
    logging.info(msg)

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    else:
        valid_loader = None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = NamingEnhancementModel(config.model)
    if config.model.ckpt_path is not None:
        model.load_state_dict(torch.load(config.model.ckpt_path)["model_state_dict"])
    model.cuda()

    criterion = get_criterion(config.train.criterion)

    optimizer, scheduler = get_optimizer_scheduler(model,
                                                   config.train.optimizer,
                                                   config.train.scheduler if "scheduler" in config.train else None)

    if valid_loader is not None:
        valid_evaluator = Evaluator(valid_loader, config.eval.metrics, 'valid', save_path, config.eval.metric_to_save)
    else:
        valid_evaluator = None

    test_evaluator = Evaluator(test_loader, config.eval.metrics, 'test', save_path, config.eval.metric_to_save)


    trainer = Trainer(model, optimizer, criterion, scheduler, train_loader, valid_evaluator, test_evaluator, config.train, config.eval)
    trainer.train()

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mit5k_dpe_config.yaml')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config)
