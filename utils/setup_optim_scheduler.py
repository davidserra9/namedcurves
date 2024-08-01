import torch
from torch.optim import lr_scheduler

def get_optimizer_scheduler(model, optim_config, scheduler_config):
    optimizer = getattr(torch.optim, optim_config.type)(model.parameters(), **optim_config.params)
    if scheduler_config is not None:
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.type)(optimizer, **scheduler_config.params)
    else:
        scheduler = None
    return optimizer, scheduler
