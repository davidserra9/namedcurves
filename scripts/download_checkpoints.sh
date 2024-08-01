#!/bin/bash

mkdir pretrained
wget https://github.com/davidserra9/namedcurves/releases/download/v1.0/mit5k_dpe_psnr_24.91.pth -P pretrained
wget https://github.com/davidserra9/namedcurves/releases/download/v1.0/mit5k_uegan_psnr_25.59.pth -P pretrained
wget https://github.com/davidserra9/namedcurves/releases/download/v1.0/ppr10k_a_psnr_26.81.pth -P pretrained
wget https://github.com/davidserra9/namedcurves/releases/download/v1.0/ppr10k_b_psnr_25.91.pth -P pretrained
wget https://github.com/davidserra9/namedcurves/releases/download/v1.0/ppr10k_c_psnr_25.69.pth -P pretrained

