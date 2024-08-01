import argparse
from models.model import NamedCurves
import torch
import os
from omegaconf import OmegaConf
from glob import glob
from PIL import Image
from torchvision.transforms import functional as TF

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='assets/a4957-input.png')
    parser.add_argument('--output_path', type=str, default='output/')
    parser.add_argument('--model_path', type=str, default='/home/dserrano/Workspace/Color-Naming-Image-Enhancement/pretrained/mit5k_uegan_psnr_25.59.pth')
    parser.add_argument('--config_path', type=str, default='configs/mit5k_dpe_config.yaml')
    return parser.parse_args()

def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)
    model = NamedCurves(config.model).cuda()
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #check if input_path is a folder
    if os.path.isdir(args.input_path):
        input_paths = glob(sorted(args.input_path + '/*'))
    
    else:
        input_paths = [args.input_path]
    
    for input_path in input_paths:
        input_tensor = TF.to_tensor(Image.open(input_path)).unsqueeze(0)
        output = model(input_tensor.cuda())
        output = TF.to_pil_image(output[0].cpu())
        output.save(os.path.join(args.output_path, os.path.basename(input_path)))

if __name__ == '__main__':
    main()