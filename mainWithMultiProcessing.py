import os.path
import numpy as np
import torch
from gym.spaces import Box
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from stable_baselines3 import DDPG, SAC
from SimpleSwerveRLEnvIntake import env

# Configuration parameters
USE_SAC = True
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
ACTION_NOISE_STD_DEV = 0.2
LAYER_SIZES = [400, 300]

# Check if CUDA is available
print(torch.cuda.is_available())

# Directory setup
models_dir = "models/SAC" if USE_SAC else "models/DDPG"
logdir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Environment initialization with multiprocessing
def make_env():
    return env()

def main():
    num_cpu = 4  # Adjust based on your CPU
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    # Initialize action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=ACTION_NOISE_STD_DEV * np.ones(n_actions))

    # Model configuration
    if USE_SAC:
        model = SAC("MlpPolicy", env, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                    device="cuda", tensorboard_log=logdir, verbose=1)
    else:
        model = DDPG("MlpPolicy", env, action_noise=action_noise, batch_size=BATCH_SIZE,
                     learning_rate=LEARNING_RATE, device="cuda", tensorboard_log=logdir, verbose=1)

    # Training
    seasons = 1
    time_steps = 100000
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

if __name__ == '__main__':
    main()