import os
import cv2
import numpy as np
from pix2pix.networks import define_G
import torch
import torchvision
import matplotlib.pyplot as plt


class Aurora_pix2pix:
    def __init__(self):
        self.model_file_path = (
            "/Users/ogawa/Desktop/desktop_folders/data/latest_net_G.pth"
        )
        self.device = torch.device("cpu")

    def load_pix2pix_generator(
        self, model_file_path: str, gpu_ids: list = [], eval: bool = False
    ):
        self.model_file_path = model_file_path
        gen = define_G(
            input_nc=1,
            output_nc=1,
            ngf=64,
            netG="unet_256",
            norm="batch",
            use_dropout=True,
            init_type="normal",
            init_gain=0.02,
            gpu_ids=[],
        )
        state_dict = torch.load(model_file_path, map_location=str(self.device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
            gen.load_state_dict(state_dict)

        if eval:
            gen.eval()
        return gen

    def translate(self, img, model: torch.nn.Module):
        with torch.no_grad():
            # 4次元テンソルに変換->ndarrayに変換
            fake = model(img.reshape((1, 1) + img.shape)).detach().cpu().numpy()
            return fake.reshape(img.shape)
