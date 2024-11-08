import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from FRCEnv import env  # Assuming this is your custom environment


# Custom Feature Extractor with 1D Convolutions
class CustomLaserFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomLaserFeatureExtractor, self).__init__(observation_space, features_dim)
        # 1D Convolution layers to handle ordered 2D laser scan data
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(2)

        # Final fully connected layer to map features to output
        self.fc = nn.Linear(128 * ((observation_space.shape[0] - 12) // 2),
                            features_dim)  # Adjusted for kernel/pooling sizes

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Assuming observations are of shape (batch_size, observation_dim)
        x = observations.unsqueeze(1)  # Add channel dimension for Conv1D
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        return torch.relu(self.fc(x))


class RLTrainer:
    def __init__(self, env, use_sac=True, use_ppo=False, learning_rate=1e-4, batch_size=256,
                 action_noise_std_dev=1.0, models_dir="models", logdir="logs"):
        self.env = env
        self.use_sac = use_sac
        self.use_ppo = use_ppo
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.action_noise_std_dev = action_noise_std_dev
        self.models_dir = models_dir
        self.logdir = logdir

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize the model
        self.model = self._init_model()

    def _init_model(self):
        n_actions = self.env.action_space.shape[-1]

        policy_kwargs = dict(
            features_extractor_class=CustomLaserFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 256]
        )

        if self.use_sac:
            # Define action noise for SAC
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.action_noise_std_dev * np.ones(n_actions)
            )

            model = SAC(
                "MlpPolicy",
                self.env,
                policy_kwargs=policy_kwargs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                action_noise=action_noise,
                device=self.device,
                tensorboard_log=self.logdir,
                verbose=1
            )

        elif self.use_ppo:
            # PPO doesn't use action noise
            model = PPO(
                "MlpPolicy",
                self.env,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                device=self.device,
                tensorboard_log=self.logdir,
                verbose=1
            )
        else:
            raise NotImplementedError("Only SAC and PPO are implemented with custom policy.")

        return model

    def train(self, total_timesteps=1_000_000, log_interval=10, save_interval=100_000):
        # Define a checkpoint callback to save the model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=save_interval,
            save_path=self.models_dir,
            name_prefix='rl_model'
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=checkpoint_callback
        )
        # Save the final model
        self.model.save(os.path.join(self.models_dir, "final_model"))

    def evaluate(self, n_eval_episodes=10):
        mean_reward, std_reward = evaluate_policy(
            self.model, self.env, n_eval_episodes=n_eval_episodes, render=False
        )
        print(f"Mean Reward: {mean_reward} +/- {std_reward}")

    def test(self, n_episodes=5):
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()
                total_reward += reward
                done = terminated or truncated
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        self.env.close()


if __name__ == "__main__":
    # Initialize environment
    environment = env()

    # Create the trainer, switch between SAC and PPO with the flags `use_sac` and `use_ppo`
    trainer = RLTrainer(
        env=environment,
        use_sac=False,  # Set this to True to use SAC
        use_ppo=True,  # Set this to True to use PPO
        learning_rate=1e-3,
        batch_size=256,
        action_noise_std_dev=1.0,  # Not used for PPO
        models_dir="models/PPO",  # Change this to "models/PPO" if using PPO
        logdir="logs/PPO"  # Change this to "logs/PPO" if using PPO
    )

    # Train the model
    trainer.train(total_timesteps=3_000_000, log_interval=1, save_interval=100_000)

    # Evaluate the model
    trainer.evaluate(n_eval_episodes=10)

    # Test the model
    trainer.test(n_episodes=5)
