import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import DDPG, SAC
from SimpleSwerveRLEnvIntake import env  # Assuming this is your custom environment

from stable_baselines3.common.evaluation import evaluate_policy

# Configuration parameters
USE_SAC = True  # Set to False to use DDPG
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
ACTION_NOISE_STD_DEV = 1 # 0.2

# Check if CUDA is available
print(torch.cuda.is_available())

models_dir = "models/SAC" if USE_SAC else "models/DDPG"
logdir = "logs"

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Initialize environment
environment = env()
n_actions = environment.action_space.shape[-1]

# Define action noise for DDPG
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=ACTION_NOISE_STD_DEV * np.ones(n_actions))

# Select and configure model
if USE_SAC:
    model = SAC("MlpPolicy", environment, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                device="cuda", tensorboard_log=logdir, verbose=1)
else:
    model = DDPG("MlpPolicy", environment, action_noise=action_noise, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, device="cuda", tensorboard_log=logdir, verbose=1)

seasons = 3
time_steps = 1000000

# Training
for season in range(seasons):
    model.learn(total_timesteps=time_steps, log_interval=10)  # Log stats every 10 calls
    model.save(os.path.join(models_dir, f"Season_{season}"))  # Changed space to underscore for file name compatibility

# Testing the model
obs, info = environment.reset()

for _ in range(100000000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = environment.step(action)
    environment.render()
    if truncated or terminated:
        obs, info = environment.reset()
        # Overwrite the console line to keep the output clean and only show the most recent reset
        print("\rResetting... Last Reward: {:.2f}".format(reward), end='', flush=True)

environment.close()
