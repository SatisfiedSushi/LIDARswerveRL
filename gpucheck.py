# import torch

# if torch.cuda.is_available():
#     print("GPU is available.")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
#     print(f"GPU Name: {torch.cuda.get_device_name(0)}")
# else:
#     print("GPU is not available. Training will proceed on CPU.")

# ray_gpu_check.py

# import ray

# ray.init(ignore_reinit_error=True)

# resources = ray.available_resources()
# print(f"Available resources: {resources}")

# tensorflow gpu check

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))