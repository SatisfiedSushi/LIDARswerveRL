import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

# Import the updated environment
from FRCEnv import env  # Assuming this is your custom environment


# Custom Feature Extractor implementing the specified architecture
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        # Define observation sizes
        self.num_lidar_inputs = 400  # Total rays from both cameras
        self.num_robot_state_inputs = 9   # First 9 observations
        self.num_additional_inputs = 4    # Last 4 observations

        # Calculate indices for splitting observations
        self.lidar_start = self.num_robot_state_inputs
        self.lidar_end = self.lidar_start + self.num_lidar_inputs
        self.additional_start = self.lidar_end

        # LiDAR Input processing
        self.lidar_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=7, stride=1),
            nn.Flatten()
        )

        # Calculate LiDAR features size
        lidar_output_size = self._calculate_lidar_output_size(self.num_lidar_inputs)

        # LiDAR Fully Connected layers
        self.lidar_fc = nn.Sequential(
            nn.Linear(lidar_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Robot State Input processing
        self.robot_state_net = nn.Sequential(
            nn.Linear(self.num_robot_state_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Additional Input processing
        self.additional_net = nn.Sequential(
            nn.Linear(self.num_additional_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Combined network after concatenation
        combined_input_size = 256 + 64 + 64  # Outputs from lidar_fc, robot_state_net, additional_net

        self.combined_net = nn.Sequential(
            nn.Linear(combined_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Set the features dimension (output size of combined_net)
        self._features_dim = 128

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights using Kaiming Uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _calculate_lidar_output_size(self, input_size):
        # Calculate the output size after convolutional and pooling layers
        size = input_size
        # Conv1d layer 1
        size = ((size + 2 * 3 - 7) // 1) + 1  # padding=3, kernel_size=7, stride=1
        # MaxPool1d layer
        size = ((size - 1 * (3 - 1) - 1) // 1) + 1  # kernel_size=3, stride=1
        # Conv1d layers 2-4
        for _ in range(3):
            size = ((size + 2 * 1 - 3) // 1) + 1  # padding=1, kernel_size=3, stride=1
        # AvgPool1d layer
        size = ((size - 1 * (7 - 1) - 1) // 1) + 1  # kernel_size=7, stride=1
        output_size = 64 * size  # 64 channels from the last Conv1d layer
        return output_size

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations
        robot_state_input = observations[:, :self.num_robot_state_inputs]
        lidar_input = observations[:, self.lidar_start:self.lidar_end]
        additional_input = observations[:, self.additional_start:]

        # Process LiDAR Input
        lidar_input = lidar_input.unsqueeze(1)  # Add channel dimension
        lidar_features = self.lidar_net(lidar_input)
        lidar_features = self.lidar_fc(lidar_features)

        # Process Robot State Input
        robot_state_features = self.robot_state_net(robot_state_input)

        # Process Additional Input
        additional_features = self.additional_net(additional_input)

        # Concatenate features
        combined_features = torch.cat((lidar_features, robot_state_features, additional_features), dim=1)

        # Pass through combined network
        combined_features = self.combined_net(combined_features)

        return combined_features


# Custom Policy that uses the CustomCombinedExtractor
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        kwargs.pop('net_arch', None)
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=nn.ReLU,
            **kwargs
        )

        # Replace the features extractor with our custom one
        self.features_extractor = CustomCombinedExtractor(observation_space)

        # The output of the features extractor is of size 128
        features_dim = self.features_extractor._features_dim

        # Action network
        self.action_net = nn.Linear(features_dim, action_space.shape[0])

        # Value network
        self.value_net = nn.Linear(features_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights using Kaiming Uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class RLTrainer:
    def __init__(self, env, use_sac=True, use_ppo=False, learning_rate=1e-4, batch_size=256,
                 models_dir="models", logdir="logs"):
        self.env = env
        self.use_sac = use_sac
        self.use_ppo = use_ppo
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
            net_arch=[],  # No additional layers after features extractor
        )

        if self.use_sac:
            model = SAC(
                policy=CustomActorCriticPolicy,
                env=self.env,
                policy_kwargs=policy_kwargs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                device=self.device,
                tensorboard_log=self.logdir,
                verbose=1
            )
        elif self.use_ppo:
            model = PPO(
                policy=CustomActorCriticPolicy,
                env=self.env,
                policy_kwargs=policy_kwargs,
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
        models_dir="models/PPO",  # Change this as needed
        logdir="logs/PPO"  # Change this as needed
    )

    # Train the model
    trainer.train(total_timesteps=3_000_000, log_interval=1, save_interval=100_000)

    # Evaluate the model
    trainer.evaluate(n_eval_episodes=10)

    # Test the model
    trainer.test(n_episodes=5)
