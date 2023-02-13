# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionDepth2ImgPipeline
import torch
import os
def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth",torch_dtype=torch.float16)
    

if __name__ == "__main__":
    download_model()
