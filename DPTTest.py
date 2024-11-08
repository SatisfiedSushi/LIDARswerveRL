from transformers import pipeline
from PIL import Image
import requests
from tqdm import tqdm
import os
from huggingface_hub import hf_hub_download

# Show progress for downloading the model
def download_with_progress(repo_id, filename, cache_dir='./dpt-model'):
    with tqdm(total=100, desc=f"Downloading {filename}", unit='%', leave=False) as pbar:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        pbar.update(100)  # Update to 100% when download is complete
    return file_path

# Download the model files with progress tracking
model_file = download_with_progress(repo_id="Intel/dpt-large", filename="pytorch_model.bin")
config_file = download_with_progress(repo_id="Intel/dpt-large", filename="config.json")

# Load pipeline locally using the cached model
pipe = pipeline(task="depth-estimation", model="Intel/dpt-large", cache_dir="./dpt-model")

# Load the image from the web
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Perform inference
depth = pipe(image)["depth"]

# Display depth result (optional)
print(depth)
