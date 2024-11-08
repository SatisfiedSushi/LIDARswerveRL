from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
import os

# Specify the directory where the model will be saved
model_dir = "/models/llama3.2-1b"

# Check if the model is already downloaded
if not os.path.exists(model_dir):
    print("Downloading the model...")

    # Tokenizer download with progress bar
    tokenizer = LlamaTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B", cache_dir=model_dir
    )

    # Model download with progress bar
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        cache_dir=model_dir,
        progress_bar=True  # This will show download progress
    )

    print(f"Model downloaded and saved at: {model_dir}")

else:
    print(f"Model is already downloaded and stored at: {model_dir}")
