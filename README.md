![thumbnail](/assets/thumbnail.png)

This repository is the official implementation of "NamedCurves: Learned Image Enhancement via Color Naming" @ ECCV24.

[![arXiv](https://img.shields.io/badge/ArXiv-Paper-B31B1B)](https://arxiv.org/abs/2407.09892)
[![web](https://img.shields.io/badge/Project-Page-orange)](https://namedcurves.github.io/)

[David Serrano-Lozano](https://davidserra9.github.io/), [Luis Herranz](http://www.lherranz.org/), [Michael S. Brown](http://www.cse.yorku.ca/~mbrown/) and [Javier Vazquez-Corral](https://www.jvazquez-corral.net/)

## TODO:
- Create notebook
- Create demo

## Method

We propose NamedCurves, a learning-based image enhancement technique that decomposes the image into a small set of named colors. Our method learns to globally adjust the image for each specific named color via tone curves and then combines the images using and attention-based fusion mechanism to mimic spatial editing.

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

- We provide a conda env file that contains all the required dependencies.

```
conda env create -f environment.yaml
```

- Following this, you can activte the conda environment with the command below.
```
conda activate namedcurves
```

- Or use virtual environment:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pre-trained models

Create and store the pre-trained models in a folder inside the repository.

```
cd namedcurves
mkdir pretrained
```

The weights can be found [here].

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


