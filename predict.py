import os
from cog import BasePredictor, Input, Path
from typing import List
import sys, shutil
sys.path.append('/content/TripoSR-hf')
os.chdir('/content/TripoSR-hf')

import logging
import os
import tempfile
import time

import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

rembg_session = rembg.new_session()

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image

def generate(image, model):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes)[0]
    mesh = to_gradio_3d_orientation(mesh)
    mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    mesh.export(mesh_path.name)
    return mesh_path.name

def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name = generate(preprocessed)
    return preprocessed, mesh_name

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.model = TSR.from_pretrained("/content/model", config_name="config.yaml", weight_name="model.ckpt")
        self.model.renderer.set_chunk_size(0)
        self.model.to(device)
    def predict(
        self,
        image_path: Path = Input(description="Input Image"),
        do_remove_background: bool = Input(default=True),
        foreground_ratio: float = Input(default=0.85, ge=0.5, le=1.0),
    ) -> Path:
        # check_input_image(image_path)
        image = Image.open(image_path)
        processed_image = preprocess(image, do_remove_background, foreground_ratio)
        output_model = generate(processed_image, self.model)
        return Path(output_model)