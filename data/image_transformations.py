import random
import torchvision.transforms.functional as F
from torchvision import transforms

class RandomCropPair:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2):
        i, j, h, w = transforms.RandomCrop.get_params(img1, self.size)
        img1 = F.crop(img1, i, j, h, w)
        img2 = F.crop(img2, i, j, h, w)
        return img1, img2

class ResizePair:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2):
        # antialias=True is used to avoid torchvision warning
        img1 = F.resize(img1, self.size, antialias=True)
        img2 = F.resize(img2, self.size, antialias=True)
        return img1, img2

class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
        return img1, img2

class RandomVerticalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            img1 = F.vflip(img1)
            img2 = F.vflip(img2)
        return img1, img2

def get_transforms(transforms_config):
    transform_list = []
    for transform in transforms_config:
        transform_type = transform['type']
        params = transform['params']
        if transform_type == 'RandomCrop':
            transform_list.append(RandomCropPair(**params))
        elif transform_type == 'Resize':
            transform_list.append(ResizePair(**params))
        elif transform_type == 'RandomHorizontalFlip':
            transform_list.append(RandomHorizontalFlipPair(**params))
        elif transform_type == 'RandomVerticalFlip':
            transform_list.append(RandomVerticalFlipPair(**params))
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")

    return transform_list