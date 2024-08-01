import torch
import argparse
import os.path
from PIL import Image
from models.color_naming import ColorNaming
from torchvision.transforms import functional as TF

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_categories', type=int, default=6)
    parser.add_argument('--image_path', type=str, default='/home/dserrano/Documents/datasets/FiveK-DPE/input/a0001-jmac_DSC1459.png')
    parser.add_argument('--output_path', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    color_naming = ColorNaming(num_categories=args.num_categories)

    if os.path.isfile(args.image_path):
        image_tensor = TF.pil_to_tensor(Image.open(args.image_path).convert('RGB')).unsqueeze(0)
        cn_probs = color_naming(image_tensor).float().repeat(1, 3, 1, 1).cpu()
        output_images = (1 - cn_probs) * 255 * torch.ones_like(image_tensor).repeat(args.num_categories, 1, 1, 1) + cn_probs * image_tensor.repeat(args.num_categories, 1, 1, 1)

        import matplotlib.pyplot as plt
        fig = plt.subplots(1, args.num_categories, figsize=(20, 20))
        for i in range(args.num_categories):
            plt.subplot(1, args.num_categories, i+1)
            plt.imshow(output_images[i].permute(1, 2, 0).numpy().astype('uint8'))
        plt.show()