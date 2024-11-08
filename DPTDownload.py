from huggingface_hub import hf_hub_download

print("Downloading the DPT model...")
# Download the model weights
model_path = hf_hub_download(repo_id="Intel/dpt-large", filename="pytorch_model.bin")

# Download the config and feature extractor files
print("Downloading the DPT configuration...")
config_path = hf_hub_download(repo_id="Intel/dpt-large", filename="config.json")
print("Downloading the DPT feature extractor...")
feature_extractor_path = hf_hub_download(repo_id="Intel/dpt-large", filename="preprocessor_config.json")
