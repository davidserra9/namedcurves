![thumbnail](/assets/thumbnail.png)

This repository is the official implementation of "NamedCurves: Learned Image Enhancement via Color Naming" @ ECCV24.

[![arXiv](https://img.shields.io/badge/ArXiv-Paper-B31B1B)](https://arxiv.org/abs/2407.09892)
[![web](https://img.shields.io/badge/Project-Page-orange)](https://namedcurves.github.io/)

[David Serrano-Lozano](https://davidserra9.github.io/), [Luis Herranz](http://www.lherranz.org/), [Michael S. Brown](http://www.cse.yorku.ca/~mbrown/) and [Javier Vazquez-Corral](https://www.jvazquez-corral.net/)

## News 🚀
- [Nov24]  Code update. We release the inference images for MIT5K-UEGAN.
- [July24] We realease the code and pretrained models of our paper.
- [July24] Our paper NamedCurves is accepted to ECCV24!

## TODO:
- torch Dataset object for PPR10K
- Create notebook
- Create gradio demo

## Method

We propose NamedCurves, a learning-based image enhancement technique that decomposes the image into a small set of named colors. Our method learns to globally adjust the image for each specific named color via tone curves and then combines the images using and attention-based fusion mechanism to mimic spatial editing. In contrast to other SOTA methods, NamedCurves allows interpretability thanks to computing a set of tone curves for each universal color name. 

![architecture](/assets/architecture-overview.png)

## Data

In this paper we use two datasets: [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) and [PPR10K](https://github.com/csjliang/PPR10K).

### MIT-Adobe FiveK

MIT FiveK dataset consists of 5,000 photographs taken by SLR cameras by a set of different photographers that cover a broad range of scenes, subjects, and lighting conditions. They are all in RAW format. Then, 5 different photography students adjust the tone of the photos. Each of them retouched all the 5,000 photos using Adobe Lightroom.

Following previous works we decided to use just the expert-C redition. To obtain the retouched images, we have to render the RAW files using Adobe Lightroom. Because of this, researchers have created different rendered versions of the dataset. In this paper, we use 3 different versions: DPE, UPE and UEGAN, dubbed after the method that introduced them. Some methods were evaluated in only some of the versions and their code and models are not available, so we considered it was fair to compare our results in the same conditions as they did. Now, we will provide information on the properties of each version and how to obtain them:

The dataset can be downloaded [here](ttps://data.csail.mit.edu/graphics/fivek/). After downloading the images you will need to use Adobe Lightroom to pre-process them according to each version.

- The **DPE** version uses the first 2,250 images of the dataset for training, the following 2,250 for validation and the last 500 for testing. The images are rendered to have the short edge to 512 pixels. Please see the [issue](https://github.com/sjmoran/CURL/issues/20) for detailed instructions.

- The **UPE** version uses the first 4,500 images of the dataset for training and the last 500 for testing. The images are rendered to have the short edge to 512 pixels. Please see the [issue](https://github.com/dvlab-research/DeepUPE/issues/26) for detailed instructions.

- The **UEGAN** version uses the first 4,500 images of the dataset for training and the last 500 for testing. The images are rendered to have the short edge to 512 pixels. For downloading the rendered images from [Google Drive](https://drive.google.com/drive/folders/1x-DcqFVoxprzM4KYGl8SUif8sV-57FP3). Please see the [official repository](https://github.com/dvlab-research/DeepUPE) for more information.

### PPR10K
PPR10K contains 1,681 high-quality RAW portraits photos manually retouched by 3 experts. The dataset can be downloaded from the [official repository](https://github.com/csjliang/PPR10K). We used the 480p images.

## Getting started

### Environment setup

We provide a Conda environment file ```requirements.txt``` with all necessary dependencies, except for PyTorch and Torchvision. Follow the instructions below to set up the environment.

First, create and activate the Conda environment:

```
conda create -n namedcurves python=3.8
conda activate namedcurves
```

Alternatively, you can set up a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Next, install PyTorch and Torchvision with the appropriate versions based on your CUDA and driver dependencies. Visit the [Pytorch Official Page](https://pytorch.org/get-started/previous-versions/) for specific installation commands. For example:

```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Once PyTorch is installed, you can install the remaining dependencies from the ```requirements.txt``` file:

```
pip install -r requirements.txt
```

Alternatively, you can manually install the required packages:
```
pip install omegaconf matplotlib scipy scikit-image lpips torchmetrics
```

### Results

We provide our results for the MIT5K dataset in the following format: aXXXX_Y_Z.png, where XXXX is the 4-digit file ID, Y is the PSNR value, and Z is the $\Delta E2000$ color difference of the image. All numeric values are rounded to two decimal places.

|          | PSNR    | SSIM    | $\Delta E2000$ | Images  |
| :-------- | :------: | :-------: | :--------------: | :-------: |
| MIT5K    | 25.59   | 0.936   | 6.07           | [Link](https://cvcuab-my.sharepoint.com/:f:/g/personal/dserrano_cvc_uab_cat/EijObxqdogJHpNufwKKZE4ABI78-4iQnO78V2mHkzfs07A?e=tVTWAq)

### Pre-trained models

Create and store the pre-trained models in a folder inside the repository.

```
cd namedcurves
mkdir pretrained
```

The weights can be found [here](https://github.com/davidserra9/namedcurves/releases/tag/v1.0). Alternatively, you can run:

```
cd namedcurves
bash scripts/download_checkpoints.sh
```


## Inference

The following command takes an image file or a folder with images and saves the results in the specified directory.

```
python test.py --input_path assets/a4957-input.png --output_path output/ --config_path configs/mit5k_upe_config.yaml --model_path pretrained/mit5k_uegan_psnr_25.59.pth 
```

## Training

Modify the configurations of the ```configs``` folders and run the following command:

```
python train.py --config configs/mit5k_upe_config.yaml
```


