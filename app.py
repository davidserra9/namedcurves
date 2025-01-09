import argparse
from omegaconf import OmegaConf
import gradio as gr
from PIL import Image
import os
import torch
import numpy as np
import io
import yaml
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
#from gradio_imageslider import ImageSlider

## local code
from models.interactive_model import NamedCurves

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_named_curves(control_points):
    linspace = torch.linspace(0, 1, steps=101).unsqueeze(0).unsqueeze(2).repeat(1, 3, 1, 1).to(device)
    outspace = model.bcpe.apply_cubic_bezier(linspace, control_points)

    fig = plt.figure()
    plt.plot(linspace[0, 0, :, 0].cpu().numpy(),outspace[0, 0, :, 0].cpu().numpy(), 'r')
    plt.plot(linspace[0, 1, :, 0].cpu().numpy(), outspace[0, 1, :, 0].cpu().numpy(), 'g')
    plt.plot(linspace[0, 2, :, 0].cpu().numpy(), outspace[0, 2, :, 0].cpu().numpy(), 'b')

    plt.scatter(control_points[0, 0, :, 1].cpu().numpy(), control_points[0, 0, :, 0].cpu().numpy(), c='r', marker='x')
    plt.scatter(control_points[0, 1, :, 1].cpu().numpy(), control_points[0, 1, :, 0].cpu().numpy(), c='g', marker='x')
    plt.scatter(control_points[0, 2, :, 1].cpu().numpy(), control_points[0, 2, :, 0].cpu().numpy(), c='b', marker='x')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return Image.open(img_buf)

hf_hub_download(repo_id="davidserra9/NamedCurves", filename="mit5k_uegan_psnr_25.59.pth", local_dir="./")

CONFIG = "configs/mit5k_upe_config.yaml"
model_pt = "mit5k_uegan_psnr_25.59.pth"

# parse config file
config = OmegaConf.load(CONFIG)

device = "cuda" if torch.cuda.is_available() else "cpu"

config['train']['cuda_visible_device'] = device
model = NamedCurves(config.model, device=device).to(device)
model.load_state_dict(torch.load(model_pt)["model_state_dict"])

def load_img(filename, norm=True,):
    img = np.array(Image.open(filename).convert("RGB"))
    if norm:
        img = img / 255.
        img = img.astype(np.float32)
    return img

def process_img(image):
    img = np.array(image)
    img = img / 255.
    img = img.astype(np.float32)
    y = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_img, control_points = model(y, return_curves=True)
        
        img_curves = [get_named_curves(control_points_i) for control_points_i in control_points]
        
    enhanced_img = enhanced_img.squeeze().permute(1,2,0).clamp_(0, 1).cpu().detach().numpy()
    enhanced_img = np.clip(enhanced_img, 0. , 1.)

    enhanced_img = (enhanced_img * 255.0).round().astype(np.uint8)  # float32 to uint8
    oby_points = control_points[0][0, :, :, 0].detach().cpu().numpy()
    achr_points = control_points[1][0, :, :, 0].detach().cpu().numpy()
    pp_points = control_points[2][0, :, :, 0].detach().cpu().numpy()
    red_points = control_points[3][0, :, :, 0].detach().cpu().numpy()
    green_points = control_points[4][0, :, :, 0].detach().cpu().numpy()
    blue_points = control_points[5][0, :, :, 0].detach().cpu().numpy()

    return img_curves[0], oby_points[0, 0], oby_points[1, 0], oby_points[2, 0], oby_points[0, 1], oby_points[1, 1], oby_points[2, 1], oby_points[0, 2], oby_points[1, 2], oby_points[2, 2], oby_points[0, 3], oby_points[1, 3], oby_points[2, 3], oby_points[0, 4], oby_points[1, 4], oby_points[2, 4], oby_points[0, 5], oby_points[1, 5], oby_points[2, 5], oby_points[0, 6], oby_points[1, 6], oby_points[2, 6], oby_points[0, 7], oby_points[1, 7], oby_points[2, 7], oby_points[0, 8], oby_points[1, 8], oby_points[2, 8], oby_points[0, 9], oby_points[1, 9], oby_points[2, 9], oby_points[0, 10], oby_points[1, 10], oby_points[2, 10], img_curves[1], achr_points[0, 0], achr_points[1, 0], achr_points[2, 0], achr_points[0, 1], achr_points[1, 1], achr_points[2, 1], achr_points[0, 2], achr_points[1, 2], achr_points[2, 2], achr_points[0, 3], achr_points[1, 3], achr_points[2, 3], achr_points[0, 4], achr_points[1, 4], achr_points[2, 4], achr_points[0, 5], achr_points[1, 5], achr_points[2, 5], achr_points[0, 6], achr_points[1, 6], achr_points[2, 6], achr_points[0, 7], achr_points[1, 7], achr_points[2, 7], achr_points[0, 8], achr_points[1, 8], achr_points[2, 8], achr_points[0, 9], achr_points[1, 9], achr_points[2, 9], achr_points[0, 10], achr_points[1, 10], achr_points[2, 10], img_curves[2], pp_points[0, 0], pp_points[1, 0], pp_points[2, 0], pp_points[0, 1], pp_points[1, 1], pp_points[2, 1], pp_points[0, 2], pp_points[1, 2], pp_points[2, 2], pp_points[0, 3], pp_points[1, 3], pp_points[2, 3], pp_points[0, 4], pp_points[1, 4], pp_points[2, 4], pp_points[0, 5], pp_points[1, 5], pp_points[2, 5], pp_points[0, 6], pp_points[1, 6], pp_points[2, 6], pp_points[0, 7], pp_points[1, 7], pp_points[2, 7], pp_points[0, 8], pp_points[1, 8], pp_points[2, 8], pp_points[0, 9], pp_points[1, 9], pp_points[2, 9], pp_points[0, 10], pp_points[1, 10], pp_points[2, 10], img_curves[3], red_points[0, 0], red_points[1, 0], red_points[2, 0], red_points[0, 1], red_points[1, 1], red_points[2, 1], red_points[0, 2], red_points[1, 2], red_points[2, 2], red_points[0, 3], red_points[1, 3], red_points[2, 3], red_points[0, 4], red_points[1, 4], red_points[2, 4], red_points[0, 5], red_points[1, 5], red_points[2, 5], red_points[0, 6], red_points[1, 6], red_points[2, 6], red_points[0, 7], red_points[1, 7], red_points[2, 7], red_points[0, 8], red_points[1, 8], red_points[2, 8], red_points[0, 9], red_points[1, 9], red_points[2, 9], red_points[0, 10], red_points[1, 10], red_points[2, 10], img_curves[4], green_points[0, 0], green_points[1, 0], green_points[2, 0], green_points[0, 1], green_points[1, 1], green_points[2, 1], green_points[0, 2], green_points[1, 2], green_points[2, 2], green_points[0, 3], green_points[1, 3], green_points[2, 3], green_points[0, 4], green_points[1, 4], green_points[2, 4], green_points[0, 5], green_points[1, 5], green_points[2, 5], green_points[0, 6], green_points[1, 6], green_points[2, 6], green_points[0, 7], green_points[1, 7], green_points[2, 7], green_points[0, 8], green_points[1, 8], green_points[2, 8], green_points[0, 9], green_points[1, 9], green_points[2, 9], green_points[0, 10], green_points[1, 10], green_points[2, 10], img_curves[5], blue_points[0, 0], blue_points[1, 0], blue_points[2, 0], blue_points[0, 1], blue_points[1, 1], blue_points[2, 1], blue_points[0, 2], blue_points[1, 2], blue_points[2, 2], blue_points[0, 3], blue_points[1, 3], blue_points[2, 3], blue_points[0, 4], blue_points[1, 4], blue_points[2, 4], blue_points[0, 5], blue_points[1, 5], blue_points[2, 5], blue_points[0, 6], blue_points[1, 6], blue_points[2, 6], blue_points[0, 7], blue_points[1, 7], blue_points[2, 7], blue_points[0, 8], blue_points[1, 8], blue_points[2, 8], blue_points[0, 9], blue_points[1, 9], blue_points[2, 9], blue_points[0, 10], blue_points[1, 10], blue_points[2, 10], Image.fromarray(enhanced_img)

def process_img_with_sliders(image, oby_red_p0, oby_green_p0, oby_blue_p0, oby_red_p1, oby_green_p1, oby_blue_p1, oby_red_p2, oby_green_p2, oby_blue_p2, oby_red_p3, oby_green_p3, oby_blue_p3, oby_red_p4, oby_green_p4, oby_blue_p4, oby_red_p5, oby_green_p5, oby_blue_p5, oby_red_p6, oby_green_p6, oby_blue_p6, oby_red_p7, oby_green_p7, oby_blue_p7, oby_red_p8, oby_green_p8, oby_blue_p8, oby_red_p9, oby_green_p9, oby_blue_p9, oby_red_p10, oby_green_p10, oby_blue_p10,
                 achro_red_p0, achro_green_p0, achro_blue_p0, achro_red_p1, achro_green_p1, achro_blue_p1, achro_red_p2, achro_green_p2, achro_blue_p2, achro_red_p3, achro_green_p3, achro_blue_p3, achro_red_p4, achro_green_p4, achro_blue_p4, achro_red_p5, achro_green_p5, achro_blue_p5, achro_red_p6, achro_green_p6, achro_blue_p6, achro_red_p7, achro_green_p7, achro_blue_p7, achro_red_p8, achro_green_p8, achro_blue_p8, achro_red_p9, achro_green_p9, achro_blue_p9, achro_red_p10, achro_green_p10, achro_blue_p10,
                 pp_red_p0, pp_green_p0, pp_blue_p0, pp_red_p1, pp_green_p1, pp_blue_p1, pp_red_p2, pp_green_p2, pp_blue_p2, pp_red_p3, pp_green_p3, pp_blue_p3, pp_red_p4, pp_green_p4, pp_blue_p4, pp_red_p5, pp_green_p5, pp_blue_p5, pp_red_p6, pp_green_p6, pp_blue_p6, pp_red_p7, pp_green_p7, pp_blue_p7, pp_red_p8, pp_green_p8, pp_blue_p8, pp_red_p9, pp_green_p9, pp_blue_p9, pp_red_p10, pp_green_p10, pp_blue_p10,
                 red_red_p0, red_green_p0, red_blue_p0, red_red_p1, red_green_p1, red_blue_p1, red_red_p2, red_green_p2, red_blue_p2, red_red_p3, red_green_p3, red_blue_p3, red_red_p4, red_green_p4, red_blue_p4, red_red_p5, red_green_p5, red_blue_p5, red_red_p6, red_green_p6, red_blue_p6, red_red_p7, red_green_p7, red_blue_p7, red_red_p8, red_green_p8, red_blue_p8, red_red_p9, red_green_p9, red_blue_p9, red_red_p10, red_green_p10, red_blue_p10,
                 green_red_p0, green_green_p0, green_blue_p0, green_red_p1, green_green_p1, green_blue_p1, green_red_p2, green_green_p2, green_blue_p2, green_red_p3, green_green_p3, green_blue_p3, green_red_p4, green_green_p4, green_blue_p4, green_red_p5, green_green_p5, green_blue_p5, green_red_p6, green_green_p6, green_blue_p6, green_red_p7, green_green_p7, green_blue_p7, green_red_p8, green_green_p8, green_blue_p8, green_red_p9, green_green_p9, green_blue_p9, green_red_p10, green_green_p10, green_blue_p10,
                 blue_red_p0, blue_green_p0, blue_blue_p0, blue_red_p1, blue_green_p1, blue_blue_p1, blue_red_p2, blue_green_p2, blue_blue_p2, blue_red_p3, blue_green_p3, blue_blue_p3, blue_red_p4, blue_green_p4, blue_blue_p4, blue_red_p5, blue_green_p5, blue_blue_p5, blue_red_p6, blue_green_p6, blue_blue_p6, blue_red_p7, blue_green_p7, blue_blue_p7, blue_red_p8, blue_green_p8, blue_blue_p8, blue_red_p9, blue_green_p9, blue_blue_p9, blue_red_p10, blue_green_p10, blue_blue_p10,
                 ):
    
    x = np.linspace(0, 1, 11)
    oby_r_y = [float(oby_red_p0), float(oby_red_p1), float(oby_red_p2), float(oby_red_p3), float(oby_red_p4), float(oby_red_p5), float(oby_red_p6), float(oby_red_p7), float(oby_red_p8), float(oby_red_p9), float(oby_red_p10)]
    oby_g_y = [float(oby_green_p0), float(oby_green_p1), float(oby_green_p2), float(oby_green_p3), float(oby_green_p4), float(oby_green_p5), float(oby_green_p6), float(oby_green_p7), float(oby_green_p8), float(oby_green_p9), float(oby_green_p10)]
    oby_b_y = [float(oby_blue_p0), float(oby_blue_p1), float(oby_blue_p2), float(oby_blue_p3), float(oby_blue_p4), float(oby_blue_p5), float(oby_blue_p6), float(oby_blue_p7), float(oby_blue_p8), float(oby_blue_p9), float(oby_blue_p10)]
    achro_r_y = [float(achro_red_p0), float(achro_red_p1), float(achro_red_p2), float(achro_red_p3), float(achro_red_p4), float(achro_red_p5), float(achro_red_p6), float(achro_red_p7), float(achro_red_p8), float(achro_red_p9), float(achro_red_p10)]
    achro_g_y = [float(achro_green_p0), float(achro_green_p1), float(achro_green_p2), float(achro_green_p3), float(achro_green_p4), float(achro_green_p5), float(achro_green_p6), float(achro_green_p7), float(achro_green_p8), float(achro_green_p9), float(achro_green_p10)]
    achro_b_y = [float(achro_blue_p0), float(achro_blue_p1), float(achro_blue_p2), float(achro_blue_p3), float(achro_blue_p4), float(achro_blue_p5), float(achro_blue_p6), float(achro_blue_p7), float(achro_blue_p8), float(achro_blue_p9), float(achro_blue_p10)]
    pp_r_y = [float(pp_red_p0), float(pp_red_p1), float(pp_red_p2), float(pp_red_p3), float(pp_red_p4), float(pp_red_p5), float(pp_red_p6), float(pp_red_p7), float(pp_red_p8), float(pp_red_p9), float(pp_red_p10)]
    pp_g_y = [float(pp_green_p0), float(pp_green_p1), float(pp_green_p2), float(pp_green_p3), float(pp_green_p4), float(pp_green_p5), float(pp_green_p6), float(pp_green_p7), float(pp_green_p8), float(pp_green_p9), float(pp_green_p10)]
    pp_b_y = [float(pp_blue_p0), float(pp_blue_p1), float(pp_blue_p2), float(pp_blue_p3), float(pp_blue_p4), float(pp_blue_p5), float(pp_blue_p6), float(pp_blue_p7), float(pp_blue_p8), float(pp_blue_p9), float(pp_blue_p10)]
    red_r_y = [float(red_red_p0), float(red_red_p1), float(red_red_p2), float(red_red_p3), float(red_red_p4), float(red_red_p5), float(red_red_p6), float(red_red_p7), float(red_red_p8), float(red_red_p9), float(red_red_p10)]
    red_g_y = [float(red_green_p0), float(red_green_p1), float(red_green_p2), float(red_green_p3), float(red_green_p4), float(red_green_p5), float(red_green_p6), float(red_green_p7), float(red_green_p8), float(red_green_p9), float(red_green_p10)]
    red_b_y = [float(red_blue_p0), float(red_blue_p1), float(red_blue_p2), float(red_blue_p3), float(red_blue_p4), float(red_blue_p5), float(red_blue_p6), float(red_blue_p7), float(red_blue_p8), float(red_blue_p9), float(red_blue_p10)]
    green_r_y = [float(green_red_p0), float(green_red_p1), float(green_red_p2), float(green_red_p3), float(green_red_p4), float(green_red_p5), float(green_red_p6), float(green_red_p7), float(green_red_p8), float(green_red_p9), float(green_red_p10)]
    green_g_y = [float(green_green_p0), float(green_green_p1), float(green_green_p2), float(green_green_p3), float(green_green_p4), float(green_green_p5), float(green_green_p6), float(green_green_p7), float(green_green_p8), float(green_green_p9), float(green_green_p10)]
    green_b_y = [float(green_blue_p0), float(green_blue_p1), float(green_blue_p2), float(green_blue_p3), float(green_blue_p4), float(green_blue_p5), float(green_blue_p6), float(green_blue_p7), float(green_blue_p8), float(green_blue_p9), float(green_blue_p10)]
    blue_r_y = [float(blue_red_p0), float(blue_red_p1), float(blue_red_p2), float(blue_red_p3), float(blue_red_p4), float(blue_red_p5), float(blue_red_p6), float(blue_red_p7), float(blue_red_p8), float(blue_red_p9), float(blue_red_p10)]
    blue_g_y = [float(blue_green_p0), float(blue_green_p1), float(blue_green_p2), float(blue_green_p3), float(blue_green_p4), float(blue_green_p5), float(blue_green_p6), float(blue_green_p7), float(blue_green_p8), float(blue_green_p9), float(blue_green_p10)]
    blue_b_y = [float(blue_blue_p0), float(blue_blue_p1), float(blue_blue_p2), float(blue_blue_p3), float(blue_blue_p4), float(blue_blue_p5), float(blue_blue_p6), float(blue_blue_p7), float(blue_blue_p8), float(blue_blue_p9), float(blue_blue_p10)]

    oby_y = torch.concatenate([torch.tensor(np.array([oby_r_y, x]).T).unsqueeze(0), torch.tensor(np.array([oby_g_y, x]).T).unsqueeze(0), torch.tensor(np.array([oby_b_y, x]).T).unsqueeze(0)], dim=0).unsqueeze(0).to(device)
    achro_y = torch.concatenate([torch.tensor(np.array([achro_r_y, x]).T).unsqueeze(0), torch.tensor(np.array([achro_g_y, x]).T).unsqueeze(0), torch.tensor(np.array([achro_b_y, x]).T).unsqueeze(0)], dim=0).unsqueeze(0).to(device)
    pp_y = torch.concatenate([torch.tensor(np.array([pp_r_y, x]).T).unsqueeze(0), torch.tensor(np.array([pp_g_y, x]).T).unsqueeze(0), torch.tensor(np.array([pp_b_y, x]).T).unsqueeze(0)], dim=0).unsqueeze(0).to(device)
    red_y = torch.concatenate([torch.tensor(np.array([red_r_y, x]).T).unsqueeze(0), torch.tensor(np.array([red_g_y, x]).T).unsqueeze(0), torch.tensor(np.array([red_b_y, x]).T).unsqueeze(0)], dim=0).unsqueeze(0).to(device)
    green_y = torch.concatenate([torch.tensor(np.array([green_r_y, x]).T).unsqueeze(0), torch.tensor(np.array([green_g_y, x]).T).unsqueeze(0), torch.tensor(np.array([green_b_y, x]).T).unsqueeze(0)], dim=0).unsqueeze(0).to(device)
    blue_y = torch.concatenate([torch.tensor(np.array([blue_r_y, x]).T).unsqueeze(0), torch.tensor(np.array([blue_g_y, x]).T).unsqueeze(0), torch.tensor(np.array([blue_b_y, x]).T).unsqueeze(0)], dim=0).unsqueeze(0).to(device)

    control_points = [oby_y, achro_y, pp_y, red_y, green_y, blue_y]

    img = np.array(image)
    img = img / 255.
    img = img.astype(np.float32)
    y = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_img, control_points = model(y, return_curves=True, control_points=control_points)
        
        img_curves = [get_named_curves(control_points_i) for control_points_i in control_points]
        
    enhanced_img = enhanced_img.squeeze().permute(1,2,0).clamp_(0, 1).cpu().detach().numpy()
    enhanced_img = np.clip(enhanced_img, 0. , 1.)

    enhanced_img = (enhanced_img * 255.0).round().astype(np.uint8)  # float32 to uint8
    oby_points = control_points[0][0, :, :, 0].detach().cpu().numpy()
    achr_points = control_points[1][0, :, :, 0].detach().cpu().numpy()
    pp_points = control_points[2][0, :, :, 0].detach().cpu().numpy()
    red_points = control_points[3][0, :, :, 0].detach().cpu().numpy()
    green_points = control_points[4][0, :, :, 0].detach().cpu().numpy()
    blue_points = control_points[5][0, :, :, 0].detach().cpu().numpy()

    return img_curves[0], oby_points[0, 0], oby_points[1, 0], oby_points[2, 0], oby_points[0, 1], oby_points[1, 1], oby_points[2, 1], oby_points[0, 2], oby_points[1, 2], oby_points[2, 2], oby_points[0, 3], oby_points[1, 3], oby_points[2, 3], oby_points[0, 4], oby_points[1, 4], oby_points[2, 4], oby_points[0, 5], oby_points[1, 5], oby_points[2, 5], oby_points[0, 6], oby_points[1, 6], oby_points[2, 6], oby_points[0, 7], oby_points[1, 7], oby_points[2, 7], oby_points[0, 8], oby_points[1, 8], oby_points[2, 8], oby_points[0, 9], oby_points[1, 9], oby_points[2, 9], oby_points[0, 10], oby_points[1, 10], oby_points[2, 10], img_curves[1], achr_points[0, 0], achr_points[1, 0], achr_points[2, 0], achr_points[0, 1], achr_points[1, 1], achr_points[2, 1], achr_points[0, 2], achr_points[1, 2], achr_points[2, 2], achr_points[0, 3], achr_points[1, 3], achr_points[2, 3], achr_points[0, 4], achr_points[1, 4], achr_points[2, 4], achr_points[0, 5], achr_points[1, 5], achr_points[2, 5], achr_points[0, 6], achr_points[1, 6], achr_points[2, 6], achr_points[0, 7], achr_points[1, 7], achr_points[2, 7], achr_points[0, 8], achr_points[1, 8], achr_points[2, 8], achr_points[0, 9], achr_points[1, 9], achr_points[2, 9], achr_points[0, 10], achr_points[1, 10], achr_points[2, 10], img_curves[2], pp_points[0, 0], pp_points[1, 0], pp_points[2, 0], pp_points[0, 1], pp_points[1, 1], pp_points[2, 1], pp_points[0, 2], pp_points[1, 2], pp_points[2, 2], pp_points[0, 3], pp_points[1, 3], pp_points[2, 3], pp_points[0, 4], pp_points[1, 4], pp_points[2, 4], pp_points[0, 5], pp_points[1, 5], pp_points[2, 5], pp_points[0, 6], pp_points[1, 6], pp_points[2, 6], pp_points[0, 7], pp_points[1, 7], pp_points[2, 7], pp_points[0, 8], pp_points[1, 8], pp_points[2, 8], pp_points[0, 9], pp_points[1, 9], pp_points[2, 9], pp_points[0, 10], pp_points[1, 10], pp_points[2, 10], img_curves[3], red_points[0, 0], red_points[1, 0], red_points[2, 0], red_points[0, 1], red_points[1, 1], red_points[2, 1], red_points[0, 2], red_points[1, 2], red_points[2, 2], red_points[0, 3], red_points[1, 3], red_points[2, 3], red_points[0, 4], red_points[1, 4], red_points[2, 4], red_points[0, 5], red_points[1, 5], red_points[2, 5], red_points[0, 6], red_points[1, 6], red_points[2, 6], red_points[0, 7], red_points[1, 7], red_points[2, 7], red_points[0, 8], red_points[1, 8], red_points[2, 8], red_points[0, 9], red_points[1, 9], red_points[2, 9], red_points[0, 10], red_points[1, 10], red_points[2, 10], img_curves[4], green_points[0, 0], green_points[1, 0], green_points[2, 0], green_points[0, 1], green_points[1, 1], green_points[2, 1], green_points[0, 2], green_points[1, 2], green_points[2, 2], green_points[0, 3], green_points[1, 3], green_points[2, 3], green_points[0, 4], green_points[1, 4], green_points[2, 4], green_points[0, 5], green_points[1, 5], green_points[2, 5], green_points[0, 6], green_points[1, 6], green_points[2, 6], green_points[0, 7], green_points[1, 7], green_points[2, 7], green_points[0, 8], green_points[1, 8], green_points[2, 8], green_points[0, 9], green_points[1, 9], green_points[2, 9], green_points[0, 10], green_points[1, 10], green_points[2, 10], img_curves[5], blue_points[0, 0], blue_points[1, 0], blue_points[2, 0], blue_points[0, 1], blue_points[1, 1], blue_points[2, 1], blue_points[0, 2], blue_points[1, 2], blue_points[2, 2], blue_points[0, 3], blue_points[1, 3], blue_points[2, 3], blue_points[0, 4], blue_points[1, 4], blue_points[2, 4], blue_points[0, 5], blue_points[1, 5], blue_points[2, 5], blue_points[0, 6], blue_points[1, 6], blue_points[2, 6], blue_points[0, 7], blue_points[1, 7], blue_points[2, 7], blue_points[0, 8], blue_points[1, 8], blue_points[2, 8], blue_points[0, 9], blue_points[1, 9], blue_points[2, 9], blue_points[0, 10], blue_points[1, 10], blue_points[2, 10], Image.fromarray(enhanced_img)
        
        
        
        
title = "NamedCurves🌈🤗"
description = '''
'''

article = "<p style='text-align: center'><a href='https://github.com/davidserra9/namedcurves' target='_blank'>NamedCurves: Learned Image Enhancement via Color Naming</a></p>"

#### Image,Prompts examples
#examples = [['assets/a4957-input.png']]

css = """
    .image-frame img, .image-container img {
        width: auto;
        height: auto;
        max-width: none;
    }
"""

with gr.Blocks() as demo:
    gr.Markdown("""
    ## [NamedCurves](https://namedcurves.github.io/): Learned Image Enhancement via Color Naming
    [David Serrano-Lozano](https://davidserra9.github.io/), [Luis Herranz](https://www.lherranz.org/), [Michael S. Brown](https://www.eecs.yorku.ca/~mbrown/), [Javier Vazquez-Corral](https://jvazquezcorral.github.io/)
    Computer Vision Center, Universitat Autònoma de Barcelona, Universidad Autónoma de Madrid, York University
    
    **NamedCurves decomposes an image into a small set of named colors and enhances them using a learned set of tone curves.** By making this decomposition, we improve the interactivity of the model as the user can modify the tone curves assigned to color name to manipulate only a certain color of the image.
    
    * Upload an image and click "Run" to automatically enhance the image. 
    * Then, you can adjust the tone curves for each color name to make the retouched version to your liking. Manipulate the sliders corresponding to the control points that define the tone curves. Note that for simplicity, we show the intensity values of the control points instead of the RGB values of each control point.
                    
    """)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Input")
            run_btn = gr.Button("Run")
            examples = gr.Examples(['assets/a4957-input.png', 'assets/a4996-input.png', 'assets/a4998-input.png', 'assets/a5000-input.png',
                                    'assets/a4986-input.png', 'assets/a4988-input.png', 'assets/a4990-input.png', 'assets/a4993-input.png'], inputs=[image])

        with gr.Tabs() as input_tabs:
                with gr.Tab(label="Orange", id=0) as oby_curves_tab:
                    oby_curves = gr.Image(label="Orange-Brown-Yellow Curves", type="pil")
                    with gr.Tabs() as channel_tabs:
                        with gr.Tab(label="R", id=0) as oby_red_curves_tab:
                            oby_red_p0 = gr.Slider(0, 1, label="R-P0", interactive=True)
                            oby_red_p1 = gr.Slider(0, 1, label="R-P1", interactive=True)
                            oby_red_p2 = gr.Slider(0, 1, label="R-P2", interactive=True)
                            oby_red_p3 = gr.Slider(0, 1, label="R-P3", interactive=True)
                            oby_red_p4 = gr.Slider(0, 1, label="R-P4", interactive=True)
                            oby_red_p5 = gr.Slider(0, 1, label="R-P5", interactive=True)
                            oby_red_p6 = gr.Slider(0, 1, label="R-P6", interactive=True)
                            oby_red_p7 = gr.Slider(0, 1, label="R-P7", interactive=True)
                            oby_red_p8 = gr.Slider(0, 1, label="R-P8", interactive=True)
                            oby_red_p9 = gr.Slider(0, 1, label="R-P9", interactive=True)
                            oby_red_p10 = gr.Slider(0, 1, label="R-P10", interactive=True)
                        with gr.Tab(label="G", id=1) as oby_green_curves_tab:
                            oby_green_p0 = gr.Slider(0, 1, label="G-P0", interactive=True)
                            oby_green_p1 = gr.Slider(0, 1, label="G-P1", interactive=True)
                            oby_green_p2 = gr.Slider(0, 1, label="G-P2", interactive=True)
                            oby_green_p3 = gr.Slider(0, 1, label="G-P3", interactive=True)
                            oby_green_p4 = gr.Slider(0, 1, label="G-P4", interactive=True)
                            oby_green_p5 = gr.Slider(0, 1, label="G-P5", interactive=True)
                            oby_green_p6 = gr.Slider(0, 1, label="G-P6", interactive=True)
                            oby_green_p7 = gr.Slider(0, 1, label="G-P7", interactive=True)
                            oby_green_p8 = gr.Slider(0, 1, label="G-P8", interactive=True)
                            oby_green_p9 = gr.Slider(0, 1, label="G-P9", interactive=True)
                            oby_green_p10 = gr.Slider(0, 1, label="G-P10", interactive=True)
                        with gr.Tab(label="B", id=2) as oby_blue_curves_tab:
                            oby_blue_p0 = gr.Slider(0, 1, label="B-P0", interactive=True)
                            oby_blue_p1 = gr.Slider(0, 1, label="B-P1", interactive=True)
                            oby_blue_p2 = gr.Slider(0, 1, label="B-P2", interactive=True)
                            oby_blue_p3 = gr.Slider(0, 1, label="B-P3", interactive=True)
                            oby_blue_p4 = gr.Slider(0, 1, label="B-P4", interactive=True)
                            oby_blue_p5 = gr.Slider(0, 1, label="B-P5", interactive=True)
                            oby_blue_p6 = gr.Slider(0, 1, label="B-P6", interactive=True)
                            oby_blue_p7 = gr.Slider(0, 1, label="B-P7", interactive=True)
                            oby_blue_p8 = gr.Slider(0, 1, label="B-P8", interactive=True)
                            oby_blue_p9 = gr.Slider(0, 1, label="B-P9", interactive=True)
                            oby_blue_p10 = gr.Slider(0, 1, label="B-P10", interactive=True)

                with gr.Tab(label="Achr", id=1) as achro_curves_tab:
                    achro_curves = gr.Image(label="Achromatic Curves", type="pil")
                    with gr.Tabs() as channel_tabs:
                        with gr.Tab(label="R", id=0) as achro_red_curves_tab:
                            achro_red_p0 = gr.Slider(0, 1, label="R-P0", interactive=True)
                            achro_red_p1 = gr.Slider(0, 1, label="R-P1", interactive=True)
                            achro_red_p2 = gr.Slider(0, 1, label="R-P2", interactive=True)
                            achro_red_p3 = gr.Slider(0, 1, label="R-P3", interactive=True)
                            achro_red_p4 = gr.Slider(0, 1, label="R-P4", interactive=True)
                            achro_red_p5 = gr.Slider(0, 1, label="R-P5", interactive=True)
                            achro_red_p6 = gr.Slider(0, 1, label="R-P6", interactive=True)
                            achro_red_p7 = gr.Slider(0, 1, label="R-P7", interactive=True)
                            achro_red_p8 = gr.Slider(0, 1, label="R-P8", interactive=True)
                            achro_red_p9 = gr.Slider(0, 1, label="R-P9", interactive=True)
                            achro_red_p10 = gr.Slider(0, 1, label="R-P10", interactive=True)
                        with gr.Tab(label="G", id=1) as achro_green_curves_tab:
                            achro_green_p0 = gr.Slider(0, 1, label="G-P0", interactive=True)
                            achro_green_p1 = gr.Slider(0, 1, label="G-P1", interactive=True)
                            achro_green_p2 = gr.Slider(0, 1, label="G-P2", interactive=True)
                            achro_green_p3 = gr.Slider(0, 1, label="G-P3", interactive=True)
                            achro_green_p4 = gr.Slider(0, 1, label="G-P4", interactive=True)
                            achro_green_p5 = gr.Slider(0, 1, label="G-P5", interactive=True)
                            achro_green_p6 = gr.Slider(0, 1, label="G-P6", interactive=True)
                            achro_green_p7 = gr.Slider(0, 1, label="G-P7", interactive=True)
                            achro_green_p8 = gr.Slider(0, 1, label="G-P8", interactive=True)
                            achro_green_p9 = gr.Slider(0, 1, label="G-P9", interactive=True)
                            achro_green_p10 = gr.Slider(0, 1, label="G-P10", interactive=True)
                        with gr.Tab(label="B", id=2) as achro_blue_curves_tab:
                            achro_blue_p0 = gr.Slider(0, 1, label="B-P0", interactive=True)
                            achro_blue_p1 = gr.Slider(0, 1, label="B-P1", interactive=True)
                            achro_blue_p2 = gr.Slider(0, 1, label="B-P2", interactive=True)
                            achro_blue_p3 = gr.Slider(0, 1, label="B-P3", interactive=True)
                            achro_blue_p4 = gr.Slider(0, 1, label="B-P4", interactive=True)
                            achro_blue_p5 = gr.Slider(0, 1, label="B-P5", interactive=True)
                            achro_blue_p6 = gr.Slider(0, 1, label="B-P6", interactive=True)
                            achro_blue_p7 = gr.Slider(0, 1, label="B-P7", interactive=True)
                            achro_blue_p8 = gr.Slider(0, 1, label="B-P8", interactive=True)
                            achro_blue_p9 = gr.Slider(0, 1, label="B-P9", interactive=True)
                            achro_blue_p10 = gr.Slider(0, 1, label="B-P10", interactive=True)

                with gr.Tab(label="Pink", id=2) as pink_purple_curves_tab:
                    pink_purple_curves = gr.Image(label="Pink-Purple Curves", type="pil")
                    with gr.Tabs() as channel_tabs:
                        with gr.Tab(label="R", id=0) as pp_red_curves_tab:
                            pp_red_p0 = gr.Slider(0, 1, label="R-P0", interactive=True)
                            pp_red_p1 = gr.Slider(0, 1, label="R-P1", interactive=True)
                            pp_red_p2 = gr.Slider(0, 1, label="R-P2", interactive=True)
                            pp_red_p3 = gr.Slider(0, 1, label="R-P3", interactive=True)
                            pp_red_p4 = gr.Slider(0, 1, label="R-P4", interactive=True)
                            pp_red_p5 = gr.Slider(0, 1, label="R-P5", interactive=True)
                            pp_red_p6 = gr.Slider(0, 1, label="R-P6", interactive=True)
                            pp_red_p7 = gr.Slider(0, 1, label="R-P7", interactive=True)
                            pp_red_p8 = gr.Slider(0, 1, label="R-P8", interactive=True)
                            pp_red_p9 = gr.Slider(0, 1, label="R-P9", interactive=True)
                            pp_red_p10 = gr.Slider(0, 1, label="R-P10", interactive=True)
                        with gr.Tab(label="G", id=1) as pp_green_curves_tab:
                            pp_green_p0 = gr.Slider(0, 1, label="G-P0", interactive=True)
                            pp_green_p1 = gr.Slider(0, 1, label="G-P1", interactive=True)
                            pp_green_p2 = gr.Slider(0, 1, label="G-P2", interactive=True)
                            pp_green_p3 = gr.Slider(0, 1, label="G-P3", interactive=True)
                            pp_green_p4 = gr.Slider(0, 1, label="G-P4", interactive=True)
                            pp_green_p5 = gr.Slider(0, 1, label="G-P5", interactive=True)
                            pp_green_p6 = gr.Slider(0, 1, label="G-P6", interactive=True)
                            pp_green_p7 = gr.Slider(0, 1, label="G-P7", interactive=True)
                            pp_green_p8 = gr.Slider(0, 1, label="G-P8", interactive=True)
                            pp_green_p9 = gr.Slider(0, 1, label="G-P9", interactive=True)
                            pp_green_p10 = gr.Slider(0, 1, label="G-P10", interactive=True)
                        with gr.Tab(label="B", id=2) as pp_blue_curves_tab:
                            pp_blue_p0 = gr.Slider(0, 1, label="B-P0", interactive=True)
                            pp_blue_p1 = gr.Slider(0, 1, label="B-P1", interactive=True)
                            pp_blue_p2 = gr.Slider(0, 1, label="B-P2", interactive=True)
                            pp_blue_p3 = gr.Slider(0, 1, label="B-P3", interactive=True)
                            pp_blue_p4 = gr.Slider(0, 1, label="B-P4", interactive=True)
                            pp_blue_p5 = gr.Slider(0, 1, label="B-P5", interactive=True)
                            pp_blue_p6 = gr.Slider(0, 1, label="B-P6", interactive=True)
                            pp_blue_p7 = gr.Slider(0, 1, label="B-P7", interactive=True)
                            pp_blue_p8 = gr.Slider(0, 1, label="B-P8", interactive=True)
                            pp_blue_p9 = gr.Slider(0, 1, label="B-P9", interactive=True)
                            pp_blue_p10 = gr.Slider(0, 1, label="B-P10", interactive=True)

                with gr.Tab(label="Red", id=3) as red_curves_tab:
                    red_curves = gr.Image(label="Red Curves", type="pil")
                    with gr.Tabs() as channel_tabs:
                        with gr.Tab(label="R", id=0) as red_red_curves_tab:
                            red_red_p0 = gr.Slider(0, 1, label="R-P0", interactive=True)
                            red_red_p1 = gr.Slider(0, 1, label="R-P1", interactive=True)
                            red_red_p2 = gr.Slider(0, 1, label="R-P2", interactive=True)
                            red_red_p3 = gr.Slider(0, 1, label="R-P3", interactive=True)
                            red_red_p4 = gr.Slider(0, 1, label="R-P4", interactive=True)
                            red_red_p5 = gr.Slider(0, 1, label="R-P5", interactive=True)
                            red_red_p6 = gr.Slider(0, 1, label="R-P6", interactive=True)
                            red_red_p7 = gr.Slider(0, 1, label="R-P7", interactive=True)
                            red_red_p8 = gr.Slider(0, 1, label="R-P8", interactive=True)
                            red_red_p9 = gr.Slider(0, 1, label="R-P9", interactive=True)
                            red_red_p10 = gr.Slider(0, 1, label="R-P10", interactive=True)
                        with gr.Tab(label="G", id=1) as red_green_curves_tab:
                            red_green_p0 = gr.Slider(0, 1, label="G-P0", interactive=True)
                            red_green_p1 = gr.Slider(0, 1, label="G-P1", interactive=True)
                            red_green_p2 = gr.Slider(0, 1, label="G-P2", interactive=True)
                            red_green_p3 = gr.Slider(0, 1, label="G-P3", interactive=True)
                            red_green_p4 = gr.Slider(0, 1, label="G-P4", interactive=True)
                            red_green_p5 = gr.Slider(0, 1, label="G-P5", interactive=True)
                            red_green_p6 = gr.Slider(0, 1, label="G-P6", interactive=True)
                            red_green_p7 = gr.Slider(0, 1, label="G-P7", interactive=True)
                            red_green_p8 = gr.Slider(0, 1, label="G-P8", interactive=True)
                            red_green_p9 = gr.Slider(0, 1, label="G-P9", interactive=True)
                            red_green_p10 = gr.Slider(0, 1, label="G-P10", interactive=True)
                        with gr.Tab(label="B", id=2) as red_blue_curves_tab:
                            red_blue_p0 = gr.Slider(0, 1, label="B-P0", interactive=True)
                            red_blue_p1 = gr.Slider(0, 1, label="B-P1", interactive=True)
                            red_blue_p2 = gr.Slider(0, 1, label="B-P2", interactive=True)
                            red_blue_p3 = gr.Slider(0, 1, label="B-P3", interactive=True)
                            red_blue_p4 = gr.Slider(0, 1, label="B-P4", interactive=True)
                            red_blue_p5 = gr.Slider(0, 1, label="B-P5", interactive=True)
                            red_blue_p6 = gr.Slider(0, 1, label="B-P6", interactive=True)
                            red_blue_p7 = gr.Slider(0, 1, label="B-P7", interactive=True)
                            red_blue_p8 = gr.Slider(0, 1, label="B-P8", interactive=True)
                            red_blue_p9 = gr.Slider(0, 1, label="B-P9", interactive=True)
                            red_blue_p10 = gr.Slider(0, 1, label="B-P10", interactive=True)

                with gr.Tab(label="Green", id=4) as green_curves_tab:
                    green_curves = gr.Image(label="Green Curves", type="pil")
                    with gr.Tabs() as channel_tabs:
                        with gr.Tab(label="R", id=0) as green_red_curves_tab:
                            green_red_p0 = gr.Slider(0, 1, label="R-P0", interactive=True)
                            green_red_p1 = gr.Slider(0, 1, label="R-P1", interactive=True)
                            green_red_p2 = gr.Slider(0, 1, label="R-P2", interactive=True)
                            green_red_p3 = gr.Slider(0, 1, label="R-P3", interactive=True)
                            green_red_p4 = gr.Slider(0, 1, label="R-P4", interactive=True)
                            green_red_p5 = gr.Slider(0, 1, label="R-P5", interactive=True)
                            green_red_p6 = gr.Slider(0, 1, label="R-P6", interactive=True)
                            green_red_p7 = gr.Slider(0, 1, label="R-P7", interactive=True)
                            green_red_p8 = gr.Slider(0, 1, label="R-P8", interactive=True)
                            green_red_p9 = gr.Slider(0, 1, label="R-P9", interactive=True)
                            green_red_p10 = gr.Slider(0, 1, label="R-P10", interactive=True)
                        with gr.Tab(label="G", id=1) as green_green_curves_tab:
                            green_green_p0 = gr.Slider(0, 1, label="G-P0", interactive=True)
                            green_green_p1 = gr.Slider(0, 1, label="G-P1", interactive=True)
                            green_green_p2 = gr.Slider(0, 1, label="G-P2", interactive=True)
                            green_green_p3 = gr.Slider(0, 1, label="G-P3", interactive=True)
                            green_green_p4 = gr.Slider(0, 1, label="G-P4", interactive=True)
                            green_green_p5 = gr.Slider(0, 1, label="G-P5", interactive=True)
                            green_green_p6 = gr.Slider(0, 1, label="G-P6", interactive=True)
                            green_green_p7 = gr.Slider(0, 1, label="G-P7", interactive=True)
                            green_green_p8 = gr.Slider(0, 1, label="G-P8", interactive=True)
                            green_green_p9 = gr.Slider(0, 1, label="G-P9", interactive=True)
                            green_green_p10 = gr.Slider(0, 1, label="G-P10", interactive=True)
                        with gr.Tab(label="B", id=2) as green_blue_curves_tab:
                            green_blue_p0 = gr.Slider(0, 1, label="B-P0", interactive=True)
                            green_blue_p1 = gr.Slider(0, 1, label="B-P1", interactive=True)
                            green_blue_p2 = gr.Slider(0, 1, label="B-P2", interactive=True)
                            green_blue_p3 = gr.Slider(0, 1, label="B-P3", interactive=True)
                            green_blue_p4 = gr.Slider(0, 1, label="B-P4", interactive=True)
                            green_blue_p5 = gr.Slider(0, 1, label="B-P5", interactive=True)
                            green_blue_p6 = gr.Slider(0, 1, label="B-P6", interactive=True)
                            green_blue_p7 = gr.Slider(0, 1, label="B-P7", interactive=True)
                            green_blue_p8 = gr.Slider(0, 1, label="B-P8", interactive=True)
                            green_blue_p9 = gr.Slider(0, 1, label="B-P9", interactive=True)
                            green_blue_p10 = gr.Slider(0, 1, label="B-P10", interactive=True)

                with gr.Tab(label="Blue", id=5) as blue_curves_tab:
                    blue_curves = gr.Image(label="Blue Curves", type="pil")
                    with gr.Tabs() as channel_tabs:
                        with gr.Tab(label="R", id=0) as blue_red_curves_tab:
                            blue_red_p0 = gr.Slider(0, 1, label="R-P0", interactive=True)
                            blue_red_p1 = gr.Slider(0, 1, label="R-P1", interactive=True)
                            blue_red_p2 = gr.Slider(0, 1, label="R-P2", interactive=True)
                            blue_red_p3 = gr.Slider(0, 1, label="R-P3", interactive=True)
                            blue_red_p4 = gr.Slider(0, 1, label="R-P4", interactive=True)
                            blue_red_p5 = gr.Slider(0, 1, label="R-P5", interactive=True)
                            blue_red_p6 = gr.Slider(0, 1, label="R-P6", interactive=True)
                            blue_red_p7 = gr.Slider(0, 1, label="R-P7", interactive=True)
                            blue_red_p8 = gr.Slider(0, 1, label="R-P8", interactive=True)
                            blue_red_p9 = gr.Slider(0, 1, label="R-P9", interactive=True)
                            blue_red_p10 = gr.Slider(0, 1, label="R-P10", interactive=True)
                        with gr.Tab(label="G", id=1) as blue_green_curves_tab:
                            blue_green_p0 = gr.Slider(0, 1, label="G-P0", interactive=True)
                            blue_green_p1 = gr.Slider(0, 1, label="G-P1", interactive=True)
                            blue_green_p2 = gr.Slider(0, 1, label="G-P2", interactive=True)
                            blue_green_p3 = gr.Slider(0, 1, label="G-P3", interactive=True)
                            blue_green_p4 = gr.Slider(0, 1, label="G-P4", interactive=True)
                            blue_green_p5 = gr.Slider(0, 1, label="G-P5", interactive=True)
                            blue_green_p6 = gr.Slider(0, 1, label="G-P6", interactive=True)
                            blue_green_p7 = gr.Slider(0, 1, label="G-P7", interactive=True)
                            blue_green_p8 = gr.Slider(0, 1, label="G-P8", interactive=True)
                            blue_green_p9 = gr.Slider(0, 1, label="G-P9", interactive=True)
                            blue_green_p10 = gr.Slider(0, 1, label="G-P10", interactive=True)
                        with gr.Tab(label="B", id=2) as blue_blue_curves_tab:
                            blue_blue_p0 = gr.Slider(0, 1, label="B-P0", interactive=True)
                            blue_blue_p1 = gr.Slider(0, 1, label="B-P1", interactive=True)
                            blue_blue_p2 = gr.Slider(0, 1, label="B-P2", interactive=True)
                            blue_blue_p3 = gr.Slider(0, 1, label="B-P3", interactive=True)
                            blue_blue_p4 = gr.Slider(0, 1, label="B-P4", interactive=True)
                            blue_blue_p5 = gr.Slider(0, 1, label="B-P5", interactive=True)
                            blue_blue_p6 = gr.Slider(0, 1, label="B-P6", interactive=True)
                            blue_blue_p7 = gr.Slider(0, 1, label="B-P7", interactive=True)
                            blue_blue_p8 = gr.Slider(0, 1, label="B-P8", interactive=True)
                            blue_blue_p9 = gr.Slider(0, 1, label="B-P9", interactive=True)
                            blue_blue_p10 = gr.Slider(0, 1, label="B-P10", interactive=True)

        with gr.Column():
            out_image = gr.Image(type="pil", label="Output")
            recompute_btn = gr.Button("Recompute")


    run_btn.click(
        process_img,
        inputs=[image],
        outputs=[oby_curves, oby_red_p0, oby_green_p0, oby_blue_p0, oby_red_p1, oby_green_p1, oby_blue_p1, oby_red_p2, oby_green_p2, oby_blue_p2, oby_red_p3, oby_green_p3, oby_blue_p3, oby_red_p4, oby_green_p4, oby_blue_p4, oby_red_p5, oby_green_p5, oby_blue_p5, oby_red_p6, oby_green_p6, oby_blue_p6, oby_red_p7, oby_green_p7, oby_blue_p7, oby_red_p8, oby_green_p8, oby_blue_p8, oby_red_p9, oby_green_p9, oby_blue_p9, oby_red_p10, oby_green_p10, oby_blue_p10,
                 achro_curves, achro_red_p0, achro_green_p0, achro_blue_p0, achro_red_p1, achro_green_p1, achro_blue_p1, achro_red_p2, achro_green_p2, achro_blue_p2, achro_red_p3, achro_green_p3, achro_blue_p3, achro_red_p4, achro_green_p4, achro_blue_p4, achro_red_p5, achro_green_p5, achro_blue_p5, achro_red_p6, achro_green_p6, achro_blue_p6, achro_red_p7, achro_green_p7, achro_blue_p7, achro_red_p8, achro_green_p8, achro_blue_p8, achro_red_p9, achro_green_p9, achro_blue_p9, achro_red_p10, achro_green_p10, achro_blue_p10,
                 pink_purple_curves, pp_red_p0, pp_green_p0, pp_blue_p0, pp_red_p1, pp_green_p1, pp_blue_p1, pp_red_p2, pp_green_p2, pp_blue_p2, pp_red_p3, pp_green_p3, pp_blue_p3, pp_red_p4, pp_green_p4, pp_blue_p4, pp_red_p5, pp_green_p5, pp_blue_p5, pp_red_p6, pp_green_p6, pp_blue_p6, pp_red_p7, pp_green_p7, pp_blue_p7, pp_red_p8, pp_green_p8, pp_blue_p8, pp_red_p9, pp_green_p9, pp_blue_p9, pp_red_p10, pp_green_p10, pp_blue_p10,
                 red_curves, red_red_p0, red_green_p0, red_blue_p0, red_red_p1, red_green_p1, red_blue_p1, red_red_p2, red_green_p2, red_blue_p2, red_red_p3, red_green_p3, red_blue_p3, red_red_p4, red_green_p4, red_blue_p4, red_red_p5, red_green_p5, red_blue_p5, red_red_p6, red_green_p6, red_blue_p6, red_red_p7, red_green_p7, red_blue_p7, red_red_p8, red_green_p8, red_blue_p8, red_red_p9, red_green_p9, red_blue_p9, red_red_p10, red_green_p10, red_blue_p10,
                 green_curves, green_red_p0, green_green_p0, green_blue_p0, green_red_p1, green_green_p1, green_blue_p1, green_red_p2, green_green_p2, green_blue_p2, green_red_p3, green_green_p3, green_blue_p3, green_red_p4, green_green_p4, green_blue_p4, green_red_p5, green_green_p5, green_blue_p5, green_red_p6, green_green_p6, green_blue_p6, green_red_p7, green_green_p7, green_blue_p7, green_red_p8, green_green_p8, green_blue_p8, green_red_p9, green_green_p9, green_blue_p9, green_red_p10, green_green_p10, green_blue_p10,
                 blue_curves, blue_red_p0, blue_green_p0, blue_blue_p0, blue_red_p1, blue_green_p1, blue_blue_p1, blue_red_p2, blue_green_p2, blue_blue_p2, blue_red_p3, blue_green_p3, blue_blue_p3, blue_red_p4, blue_green_p4, blue_blue_p4, blue_red_p5, blue_green_p5, blue_blue_p5, blue_red_p6, blue_green_p6, blue_blue_p6, blue_red_p7, blue_green_p7, blue_blue_p7, blue_red_p8, blue_green_p8, blue_blue_p8, blue_red_p9, blue_green_p9, blue_blue_p9, blue_red_p10, blue_green_p10, blue_blue_p10,
                 out_image],
    )

    recompute_btn.click(
        process_img_with_sliders,
        inputs=[image,
                oby_red_p0, oby_green_p0, oby_blue_p0, oby_red_p1, oby_green_p1, oby_blue_p1, oby_red_p2, oby_green_p2, oby_blue_p2, oby_red_p3, oby_green_p3, oby_blue_p3, oby_red_p4, oby_green_p4, oby_blue_p4, oby_red_p5, oby_green_p5, oby_blue_p5, oby_red_p6, oby_green_p6, oby_blue_p6, oby_red_p7, oby_green_p7, oby_blue_p7, oby_red_p8, oby_green_p8, oby_blue_p8, oby_red_p9, oby_green_p9, oby_blue_p9, oby_red_p10, oby_green_p10, oby_blue_p10,
                achro_red_p0, achro_green_p0, achro_blue_p0, achro_red_p1, achro_green_p1, achro_blue_p1, achro_red_p2, achro_green_p2, achro_blue_p2, achro_red_p3, achro_green_p3, achro_blue_p3, achro_red_p4, achro_green_p4, achro_blue_p4, achro_red_p5, achro_green_p5, achro_blue_p5, achro_red_p6, achro_green_p6, achro_blue_p6, achro_red_p7, achro_green_p7, achro_blue_p7, achro_red_p8, achro_green_p8, achro_blue_p8, achro_red_p9, achro_green_p9, achro_blue_p9, achro_red_p10, achro_green_p10, achro_blue_p10,
                pp_red_p0, pp_green_p0, pp_blue_p0, pp_red_p1, pp_green_p1, pp_blue_p1, pp_red_p2, pp_green_p2, pp_blue_p2, pp_red_p3, pp_green_p3, pp_blue_p3, pp_red_p4, pp_green_p4, pp_blue_p4, pp_red_p5, pp_green_p5, pp_blue_p5, pp_red_p6, pp_green_p6, pp_blue_p6, pp_red_p7, pp_green_p7, pp_blue_p7, pp_red_p8, pp_green_p8, pp_blue_p8, pp_red_p9, pp_green_p9, pp_blue_p9, pp_red_p10, pp_green_p10, pp_blue_p10,
                red_red_p0, red_green_p0, red_blue_p0, red_red_p1, red_green_p1, red_blue_p1, red_red_p2, red_green_p2, red_blue_p2, red_red_p3, red_green_p3, red_blue_p3, red_red_p4, red_green_p4, red_blue_p4, red_red_p5, red_green_p5, red_blue_p5, red_red_p6, red_green_p6, red_blue_p6, red_red_p7, red_green_p7, red_blue_p7, red_red_p8, red_green_p8, red_blue_p8, red_red_p9, red_green_p9, red_blue_p9, red_red_p10, red_green_p10, red_blue_p10,
                green_red_p0, green_green_p0, green_blue_p0, green_red_p1, green_green_p1, green_blue_p1, green_red_p2, green_green_p2, green_blue_p2, green_red_p3, green_green_p3, green_blue_p3, green_red_p4, green_green_p4, green_blue_p4, green_red_p5, green_green_p5, green_blue_p5, green_red_p6, green_green_p6, green_blue_p6, green_red_p7, green_green_p7, green_blue_p7, green_red_p8, green_green_p8, green_blue_p8, green_red_p9, green_green_p9, green_blue_p9, green_red_p10, green_green_p10, green_blue_p10,
                blue_red_p0, blue_green_p0, blue_blue_p0, blue_red_p1, blue_green_p1, blue_blue_p1, blue_red_p2, blue_green_p2, blue_blue_p2, blue_red_p3, blue_green_p3, blue_blue_p3, blue_red_p4, blue_green_p4, blue_blue_p4, blue_red_p5, blue_green_p5, blue_blue_p5, blue_red_p6, blue_green_p6, blue_blue_p6, blue_red_p7, blue_green_p7, blue_blue_p7, blue_red_p8, blue_green_p8, blue_blue_p8, blue_red_p9, blue_green_p9, blue_blue_p9, blue_red_p10, blue_green_p10, blue_blue_p10],
        outputs=[oby_curves, oby_red_p0, oby_green_p0, oby_blue_p0, oby_red_p1, oby_green_p1, oby_blue_p1, oby_red_p2, oby_green_p2, oby_blue_p2, oby_red_p3, oby_green_p3, oby_blue_p3, oby_red_p4, oby_green_p4, oby_blue_p4, oby_red_p5, oby_green_p5, oby_blue_p5, oby_red_p6, oby_green_p6, oby_blue_p6, oby_red_p7, oby_green_p7, oby_blue_p7, oby_red_p8, oby_green_p8, oby_blue_p8, oby_red_p9, oby_green_p9, oby_blue_p9, oby_red_p10, oby_green_p10, oby_blue_p10,
                 achro_curves, achro_red_p0, achro_green_p0, achro_blue_p0, achro_red_p1, achro_green_p1, achro_blue_p1, achro_red_p2, achro_green_p2, achro_blue_p2, achro_red_p3, achro_green_p3, achro_blue_p3, achro_red_p4, achro_green_p4, achro_blue_p4, achro_red_p5, achro_green_p5, achro_blue_p5, achro_red_p6, achro_green_p6, achro_blue_p6, achro_red_p7, achro_green_p7, achro_blue_p7, achro_red_p8, achro_green_p8, achro_blue_p8, achro_red_p9, achro_green_p9, achro_blue_p9, achro_red_p10, achro_green_p10, achro_blue_p10,
                 pink_purple_curves, pp_red_p0, pp_green_p0, pp_blue_p0, pp_red_p1, pp_green_p1, pp_blue_p1, pp_red_p2, pp_green_p2, pp_blue_p2, pp_red_p3, pp_green_p3, pp_blue_p3, pp_red_p4, pp_green_p4, pp_blue_p4, pp_red_p5, pp_green_p5, pp_blue_p5, pp_red_p6, pp_green_p6, pp_blue_p6, pp_red_p7, pp_green_p7, pp_blue_p7, pp_red_p8, pp_green_p8, pp_blue_p8, pp_red_p9, pp_green_p9, pp_blue_p9, pp_red_p10, pp_green_p10, pp_blue_p10,
                 red_curves, red_red_p0, red_green_p0, red_blue_p0, red_red_p1, red_green_p1, red_blue_p1, red_red_p2, red_green_p2, red_blue_p2, red_red_p3, red_green_p3, red_blue_p3, red_red_p4, red_green_p4, red_blue_p4, red_red_p5, red_green_p5, red_blue_p5, red_red_p6, red_green_p6, red_blue_p6, red_red_p7, red_green_p7, red_blue_p7, red_red_p8, red_green_p8, red_blue_p8, red_red_p9, red_green_p9, red_blue_p9, red_red_p10, red_green_p10, red_blue_p10,
                 green_curves, green_red_p0, green_green_p0, green_blue_p0, green_red_p1, green_green_p1, green_blue_p1, green_red_p2, green_green_p2, green_blue_p2, green_red_p3, green_green_p3, green_blue_p3, green_red_p4, green_green_p4, green_blue_p4, green_red_p5, green_green_p5, green_blue_p5, green_red_p6, green_green_p6, green_blue_p6, green_red_p7, green_green_p7, green_blue_p7, green_red_p8, green_green_p8, green_blue_p8, green_red_p9, green_green_p9, green_blue_p9, green_red_p10, green_green_p10, green_blue_p10,
                 blue_curves, blue_red_p0, blue_green_p0, blue_blue_p0, blue_red_p1, blue_green_p1, blue_blue_p1, blue_red_p2, blue_green_p2, blue_blue_p2, blue_red_p3, blue_green_p3, blue_blue_p3, blue_red_p4, blue_green_p4, blue_blue_p4, blue_red_p5, blue_green_p5, blue_blue_p5, blue_red_p6, blue_green_p6, blue_blue_p6, blue_red_p7, blue_green_p7, blue_blue_p7, blue_red_p8, blue_green_p8, blue_blue_p8, blue_red_p9, blue_green_p9, blue_blue_p9, blue_red_p10, blue_green_p10, blue_blue_p10,
                 out_image],
    )

if __name__ == "__main__":
    demo.launch()