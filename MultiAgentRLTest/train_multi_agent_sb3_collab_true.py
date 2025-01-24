# train_multi_agent_sb3_collab_true.py

import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from multi_agent_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue  # Ensure correct import path
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import traceback


# ==========================
# Debug Flag
# ==========================
DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# ==========================
# Custom Feature Extractor
# ==========================

class CTDEFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor that processes the flattened global observation.
    """
    def __init__(self, observation_space: gym.spaces.Box, global_observation_dim: int):
        """
        Initialize the feature extractor.
        
        Parameters:
        - observation_space: The observation space of the environment.
        - global_observation_dim: Dimension of the global observation.
        """
        super(CTDEFeatureExtractor, self).__init__(observation_space, features_dim=256)
        
        # Define the feature extractor for the global observation
        self.net = nn.Sequential(
            nn.Linear(global_observation_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
    def forward(self, observations):
        """
        Forward pass to extract features from observations.
        """
        debug_print(f"FeatureExtractor input shape: {observations.shape}")  # Debugging
        features = self.net(observations)
        debug_print(f"FeatureExtractor output shape: {features.shape}")  # Debugging
        return features

# ==========================
# Custom ValueNet with Debugging
# ==========================

class DebugValueNet(nn.Module):
    def __init__(self, input_dim):
        super(DebugValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        debug_print(f"ValueNet input shape: {x.shape}")  # Debugging
        x = self.net(x)
        debug_print(f"ValueNet output shape: {x.shape}")  # Debugging
        return x

# ==========================
# Custom Policy for CTDE with Debugging
# ==========================
class CTDEPolicy(ActorCriticPolicy):
    """
    Custom Policy class implementing Centralized Training with Decentralized Execution (CTDE).
    
    In this setup:
    - **Actor (Policy Network):** Treats the entire action space as a single action.
    - **Critic (Value Network):** Uses the entire observation (global state) to estimate the value function.
    
    Note: SB3 is inherently single-agent, so this is a workaround for multi-agent settings.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        """
        Initialize the CTDEPolicy.
        """
        # Safely retrieve 'features_extractor_kwargs' from kwargs
        features_extractor_kwargs = kwargs.get('features_extractor_kwargs', {})
        global_observation_dim = features_extractor_kwargs.get('global_observation_dim', None)
        
        if global_observation_dim is None:
            raise ValueError("`global_observation_dim` must be provided in `features_extractor_kwargs`.")
        
        # Initialize the parent class first
        super(CTDEPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # The features extractor has already processed the observation
        # Get the features dimension from the features extractor
        features_dim = self.features_extractor.features_dim
        debug_print(f"CTDEPolicy initialized with features_dim: {features_dim}")  # Debugging
        
        # Define the custom value_net to take the features as input
        self.value_net = DebugValueNet(features_dim).to(self.device)
        debug_print(f"Custom value_net initialized.")  # Debugging

    def _predict_values(self, obs: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        """
        Predict value estimates using the centralized critic.
        
        Parameters:
        - obs: Observation tensor.
        - actions: (Unused) Actions tensor.
        
        Returns:
        - value: Value estimates tensor.
        """
        debug_print(f"_predict_values called with obs shape: {obs.shape}")  # Debugging
        value = self.value_net(obs)
        return value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate actions by computing the log probabilities and entropy.
        Also compute the value estimates using the centralized critic.
        
        Parameters:
        - obs: Observation tensor.
        - actions: Actions tensor.
        
        Returns:
        - values: Value estimates tensor.
        - log_prob: Log probabilities of the actions.
        - entropy: Entropy of the action distribution.
        """
        debug_print(f"evaluate_actions called with obs shape: {obs.shape} and actions shape: {actions.shape}")  # Debugging
        # Forward pass through the actor to get distribution
        try:
            distribution = self._get_action_dist_from_latent(self._build_latent(obs))
        except AttributeError as e:
            debug_print(f"Error during _get_action_dist_from_latent: {e}")
            raise e
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Compute the value using the centralized critic
        values = self._predict_values(obs, actions)
        
        return values, log_prob, entropy

# ==========================
# Multi-Agent Trainer with CTDE
# ==========================

class MultiAgentCTDETrainer:
    """
    Trainer class implementing Centralized Training with Decentralized Execution (CTDE).
    """
    def __init__(self, env, model_kwargs=None, device='cpu'):
        """
        Initialize the MultiAgentCTDETrainer.
        
        Parameters:
        - env: The multi-agent environment.
        - model_kwargs: Additional keyword arguments for the PPO model.
        - device: 'cpu' or 'cuda'.
        """
        self.env = env
        self.device = device
        
        # Determine the global observation dimension
        sample_obs, _ = self.env.reset()
        global_observation_dim = self.env.observation_space.shape[0]
        debug_print(f"Global observation dimension: {global_observation_dim}")  # Debugging
        
        # Define the shared PPO model
        self.model = PPO(
            policy=CTDEPolicy,
            env=self.env,
            verbose=1,
            device=self.device,
            tensorboard_log="./tensorboard_logs/ctde_multi_agent",
            policy_kwargs={
                'features_extractor_class': CTDEFeatureExtractor,
                'features_extractor_kwargs': {
                    'global_observation_dim': global_observation_dim
                },
                'net_arch': dict(pi=[64, 64], vf=[256, 256])  # pi has hidden layers; vf matches DebugValueNet
            },
            **(model_kwargs if model_kwargs else {})
        )
        debug_print(f"PPO model initialized.")  # Debugging

    def train(self, total_timesteps=1_000_000, checkpoint_freq=100_000):
        """
        Train the PPO model with centralized critic.
        
        Parameters:
        - total_timesteps: Total timesteps for training.
        - checkpoint_freq: Frequency (in timesteps) to save checkpoints.
        """
        # Define checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="./checkpoints_ctde_multi_agent/",
            name_prefix="ppo_ctde_multi_agent",
            verbose=1
        )
        
        debug_print(f"Starting training for {total_timesteps} timesteps.")  # Debugging
        
        # Start training
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=checkpoint_callback
            )
        except Exception as e:
            debug_print(f"Error during training: {e}")
            raise e
        
        # Save the final model
        self.model.save("ppo_ctde_multi_agent_final")
        debug_print("Training completed and model saved.")  # Debugging

    def evaluate(self, episodes=5, render=True):
        """
        Evaluate the trained PPO model with decentralized execution.
        
        Parameters:
        - episodes: Number of episodes to evaluate.
        - render: Whether to render the environment.
        """
        for episode in range(1, episodes + 1):
            observation, _ = self.env.reset()
            done = False
            step = 0
            total_reward = 0.0
            tasks_completed = 0
            
            debug_print(f"Starting evaluation episode {episode}.")  # Debugging
            
            while not done:
                # Predict action using the model
                try:
                    action, _states = self.model.predict(observation, deterministic=True)
                except Exception as e:
                    debug_print(f"Error during action prediction: {e}")
                    raise e
                
                # Step the environment with the predicted action
                try:
                    observation, reward, done, info = self.env.step(action)
                except Exception as e:
                    debug_print(f"Error during environment step: {e}")
                    raise e
                
                # Accumulate rewards
                total_reward += reward
                
                if render:
                    self.env.render()
                
                step += 1
                tasks_completed = info.get("tasks_completed", 0)
            
            # Logging after each episode
            print(f"Episode {episode} finished after {step} steps.")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Tasks Completed: {tasks_completed} out of {self.env.num_tasks}")
            print("-" * 50)

# ==========================
# Main Execution
# ==========================

def main():
    # Define environment configuration
    env_config = {
        "num_robots": 4,
        "num_tasks": 6,
        "robots_per_task": 2,  # Collaborative tasks
        "field_size": 15.0,
        "max_episode_steps": 300,
        "collision_penalty": True,  # Penalize collisions
        "completion_radius": 0.5,
        "robot_radius": 0.3,
        "time_penalty": -0.1,
        "max_velocity": 1.0,
        "max_angular_velocity": 1.0,
    }

    # Create the multi-agent environment
    env = MultiRobotEnvSB3CollabTrue(**env_config)
    
    # Validate the environment
    try:
        check_env(env, warn=True)
        print("Environment validation successful.")
    except AssertionError as e:
        print(f"Environment validation failed: {e}")
        traceback.print_exc()
        return
    except Exception as e:
        print(f"An error occurred during environment validation: {e}")
        traceback.print_exc()
        return

    # Initialize the PPO model with the standard MlpPolicy
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/ctde_multi_agent",
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("PPO model initialized.")

    # Define checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints_ctde_multi_agent/",
        name_prefix="ppo_ctde_multi_agent",
        verbose=1
    )
    print("Checkpoint callback created.")

    # Start training
    try:
        print("Starting training...")
        model.learn(
            total_timesteps=10_000_000,  # Total timesteps for training
            callback=checkpoint_callback
        )
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return
    print("Training completed and model saved.")

    # Save the final model
    model.save("ppo_ctde_multi_agent_final")
    print("Final model saved as 'ppo_ctde_multi_agent_final.zip'.")

    # Evaluation parameters
    num_evaluation_episodes = 5
    render_env = True  # Set to False to disable rendering

    # Evaluate the trained model
    for episode in range(1, num_evaluation_episodes + 1):
        try:
            observation, _ = env.reset()
        except Exception as e:
            print(f"Error during environment reset: {e}")
            traceback.print_exc()
            break

        done = False
        step = 0
        total_reward = 0.0
        tasks_completed = 0
        
        print(f"\nStarting evaluation episode {episode}.")

        while not done:
            # Predict action using the trained model
            try:
                action, _states = model.predict(observation, deterministic=True)
            except Exception as e:
                print(f"Error during action prediction: {e}")
                traceback.print_exc()
                break

            # Step the environment with the predicted action
            try:
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as e:
                print(f"Error during environment step: {e}")
                traceback.print_exc()
                break

            # Accumulate rewards
            total_reward += reward

            # Render the environment
            if render_env:
                try:
                    env.render()
                except Exception as e:
                    print(f"Error during rendering: {e}")
                    traceback.print_exc()
                    break

            step += 1
            tasks_completed = info.get("tasks_completed", 0)
        
        # Logging after each episode
        print(f"Episode {episode} finished after {step} steps.")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Tasks Completed: {tasks_completed} out of {env.num_tasks}")
        print("-" * 50)

    # Close the environment
    try:
        env.close()
        print("Environment closed.")
    except Exception as e:
        print(f"Error during environment closure: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()