import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG, SAC
from sb3_contrib import ppo_recurrent
from SimpleSwerveRLEnvIntake import env  # Assuming this is your custom environment

# Configuration parameters
USE_SAC = True  # Set to False to use DDPG
ACTION_NOISE_STD_DEV = 1  # 0.2

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Set directories for model loading
models_dir = "models/SAC" if USE_SAC else "models/DDPG"

# Initialize environment
environment = env()

# Select and configure model
if USE_SAC:
    model_path = os.path.join(models_dir, "WorkingFullAutoIntake.zip")  # Change this to the desired checkpoint path
    model = SAC.load(model_path, env=environment, device="cuda")
else:
    model_path = os.path.join(models_dir, "Season_2.zip")  # Change this to the desired checkpoint path
    model = DDPG.load(model_path, env=environment, device="cuda")

# Testing the loaded model
obs, info = environment.reset()

# Run for a set number of steps
for _ in range(1000):  # Adjust the number of steps as needed
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = environment.step(action)
    environment.render()
    if truncated or terminated:
        obs, info = environment.reset()
        print("\rResetting... Last Reward: {:.2f}".format(reward), end='', flush=True)

environment.close()
