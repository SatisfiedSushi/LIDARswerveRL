import os.path
import numpy as np
from gym.spaces import Box
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym
import torch as th
from SimpleSwerveRLEnv import env
from stable_baselines3 import DDPG

models_dir = "models/DDPG"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = env()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG.load("swerve-6", env=env, verbose=1, device="cuda", tensorboard_log=logdir)
# model = DDPG("MlpPolicy", env=env, action_noise=action_noise, verbose=1, device="cuda", tensorboard_log=logdir)
# model.learn(total_timesteps=100000, progress_bar=True)
# model.save("swerve-7")


obs, info = env.reset()
for i in range(100000000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    env.render()
    if truncated or terminated:
        obs, info = env.reset()
        print("Resetting")

env.close()