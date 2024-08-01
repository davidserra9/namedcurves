import os
import logging
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_tensor
from data.image_transformations import get_transforms

class MIT5KDataset(Dataset):
    def __init__(self, input_path, target_path, img_ids_filepath, transform=None):
        self.input_path = input_path
        self.target_path = target_path
        self.transform = transform
        self.img_ids = self._read_img_ids(img_ids_filepath)
        self.data = self._create_data_list()
        if transform is not None:
            self.image_transforms = get_transforms(transform)
        else:
            self.image_transforms = None

    def _read_img_ids(self, img_ids_filepath):
        # Read the image IDs from the txt file
        with open(img_ids_filepath, 'r') as f:
            img_ids = [line.strip() for line in f.readlines()]
        return img_ids

    def _create_data_list(self):
        # Create a list of dictionaries with 'input_path', 'target_path' and 'name'
        data_list = []
        for input_file in glob(os.path.join(self.input_path, "*")):
            img_id = os.path.basename(input_file).split('-')[0]
            if img_id in self.img_ids:
                target_file = os.path.join(self.target_path, os.path.basename(os.path.basename(input_file)))
                if not os.path.exists(target_file):
                    raise FileNotFoundError(f"Target file {target_file} not found. While input file {input_file} was found.")
                data_list.append({'input_path': input_file, 'target_path': target_file, 'name': img_id})

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        input_image, target_image = self._load_image_pair(data['input_path'], data['target_path'])

        return {'input_image': input_image, 'target_image': target_image, 'name':data['name']}

    def _load_image_pair(self, img1_path, img2_path):
        img1_tensor = to_tensor(np.array(Image.open(img1_path).convert('RGB')))
        img2_tensor = to_tensor(np.array(Image.open(img2_path).convert('RGB')))

        if self.image_transforms is not None:
            for image_transform in self.image_transforms:
                img1_tensor, img2_tensor = image_transform(img1_tensor, img2_tensor)

        return img1_tensor, img2_tensor

#class PPR10KDataset(Dataset):


def get_single_dataset(type, params):
    if type == 'mit5k':
        return MIT5KDataset(**params)
    elif type == 'ppr10k':
    	# TODO:
        return PPR10KDataset(**params)
    else:
        raise ValueError(f"Unsupported dataset type: {type}")


def get_datasets(config):
    """Returns the datsaets based on the configuration file."""

    if len(config) == 2:
        train_dataset = get_single_dataset(config.train.target, config.train.params)
        test_dataset = get_single_dataset(config.test.target, config.test.params)
        return train_dataset, None, test_dataset

    elif len(config) == 3:
        train_dataset = get_single_dataset(config.train.target, config.train.params)
        val_dataset = get_single_dataset(config.valid.target, config.valid.params)
        test_dataset = get_single_dataset(config.test.target, config.test.params)
        return train_dataset, val_dataset, test_dataset

    else:
        raise ValueError("The number of datasets should be 2 (train/test) or 3 (train/valid/test).")

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("../configs/mit5k_upe_config.yaml")

    dataset = MIT5KDataset(**config.data.train.params)
    input_img, target_img, name = dataset[0]

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(input_img.squeeze().permute(1, 2, 0).numpy())
    plt.title("Input Image")
    plt.subplot(1, 2, 2)
    plt.imshow(target_img.squeeze().permute(1, 2, 0).numpy())
    plt.title("Target Image")
    plt.show()



