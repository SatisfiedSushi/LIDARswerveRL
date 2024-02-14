import os.path
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from SimpleSwerveRLEnvIntake import env

# Configuration parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
ACTION_NOISE_STD_DEV = 0.2

# Check if CUDA is available
print(torch.cuda.is_available())

models_dir = "models/DDPG"
logdir = "logs"

# Create directories if they don't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Initialize environment
env = env()
n_actions = env.action_space.shape[-1]

# Define action noise
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=ACTION_NOISE_STD_DEV * np.ones(n_actions))

# Configure DDPG model
model = DDPG("MlpPolicy", env, action_noise=action_noise, batch_size=BATCH_SIZE,
             learning_rate=LEARNING_RATE, device="cuda", tensorboard_log=logdir, verbose=1)

seasons = 1
time_steps = 1000000

# Training
for i in range(seasons):
    model.learn(total_timesteps=time_steps, progress_bar=True)
    model.save(os.path.join(models_dir, f"Season {i}"))

# Testing the model
obs, info = env.reset()
for i in range(100000000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if truncated or terminated:
        obs, info = env.reset()
        print("Resetting")

env.close()
