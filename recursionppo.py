import os
import numpy as np
import torch
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from SimpleSwerveRLEnvIntake import env  # Assuming this is your custom environment

# Configuration parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 256

# Check if CUDA is available
print(torch.cuda.is_available())


models_dir = "models/RecurrentPPO"
logdir = "logs"

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Initialize environment
environment = env()

# Select and configure model
model = RecurrentPPO("MlpLstmPolicy", environment, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                     device="cuda", tensorboard_log=logdir, verbose=1)

seasons = 3
time_steps = 1000000

# Training
for season in range(seasons):
    model.learn(total_timesteps=time_steps, log_interval=10)  # Log stats every 10 calls
    model.save(os.path.join(models_dir, f"Season_{season}"))  # Changed space to underscore for file name compatibility

# Testing the model
obs, info = environment.reset()
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

for _ in range(100000000):
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, reward, terminated, truncated, info = environment.step(action)
    environment.render()
    episode_starts = np.array([terminated or truncated])
    if terminated or truncated:
        obs, info = environment.reset()
        # Overwrite the console line to keep the output clean and only show the most recent reset
        print("\rResetting... Last Reward: {:.2f}".format(reward), end='', flush=True)

environment.close()