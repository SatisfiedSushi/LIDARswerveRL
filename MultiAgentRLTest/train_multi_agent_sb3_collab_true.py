# train_multi_agent_sb3_collab_true.py

import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from multi_robot_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue  # Ensure correct import path
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import traceback
import glob
import re
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================
# Debug Flag and Utility Function
# ==========================
DEBUG = False  # Set to True to enable debug prints

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
        super(CTDEFeatureExtractor, self).__init__(observation_space, features_dim=128)
        
        # Define the feature extractor for the global observation
        self.net = nn.Sequential(
            nn.Linear(global_observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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
        
        # No need to override value_net; use the base class's value network
        # If additional customization is needed, it can be added here without disrupting base class methods

# ==========================
# Define the Clip Range Schedule Function
# ==========================

def clip_schedule(progress_remaining):
    """
    Linearly anneal the clip_range from initial_clip to final_clip.
    
    Parameters:
    - progress_remaining: Float between 1 and 0 indicating the remaining progress.
    
    Returns:
    - Float representing the current clip_range.
    """
    initial_clip = 0.2
    final_clip = 0.05
    return final_clip + (initial_clip - final_clip) * progress_remaining

# ==========================
# Function to Find the Latest Checkpoint
# ==========================
def get_latest_checkpoint(checkpoint_dir, name_prefix):
    """
    Retrieve the latest checkpoint file from the checkpoint directory based on the step count.
    
    Parameters:
    - checkpoint_dir: Directory where checkpoints are saved.
    - name_prefix: Prefix of the checkpoint files.
    
    Returns:
    - latest_checkpoint_path: Path to the latest checkpoint file or None if no checkpoints found.
    """
    checkpoint_pattern = os.path.join(checkpoint_dir, f"{name_prefix}_*.zip")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        debug_print("No checkpoint files found.")
        return None
    
    # Extract step numbers from filenames
    step_numbers = []
    pattern = re.compile(f"{re.escape(name_prefix)}_(\d+)_steps\.zip")
    for file in checkpoint_files:
        basename = os.path.basename(file)
        match = pattern.match(basename)
        if match:
            step_number = int(match.group(1))
            step_numbers.append((step_number, file))
            debug_print(f"Found checkpoint: {basename} with step {step_number}")
    
    if not step_numbers:
        debug_print("No valid checkpoint files found after pattern matching.")
        return None
    
    # Sort by step number in descending order and return the latest
    step_numbers.sort(reverse=True)
    latest_checkpoint_path = step_numbers[0][1]
    debug_print(f"Latest checkpoint found: {latest_checkpoint_path}")
    return latest_checkpoint_path

# ==========================
# Multi-Agent Trainer with CTDE
# ==========================

class MultiAgentCTDETrainer:
    """
    Trainer class implementing Centralized Training with Decentralized Execution (CTDE).
    """
    def __init__(self, env, model_kwargs=None, device='cpu', use_vec_normalize=True):
        """
        Initialize the MultiAgentCTDETrainer.
        
        Parameters:
        - env: The multi-agent environment.
        - model_kwargs: Additional keyword arguments for the PPO model.
        - device: 'cpu' or 'cuda'.
        - use_vec_normalize: Whether to use VecNormalize for observation and reward normalization.
        """
        self.env = env
        self.device = device
        
        # Determine the global observation dimension
        sample_obs, _ = self.env.reset()
        global_observation_dim = self.env.observation_space.shape[0]
        debug_print(f"Global observation dimension: {global_observation_dim}")  # Debugging
        
        # Wrap the environment with DummyVecEnv and VecNormalize for normalization
        self.vec_env = DummyVecEnv([lambda: self.env])
        if use_vec_normalize:
            self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
            debug_print("VecNormalize applied to the environment.")
        
        # Define the shared PPO model
        self.model = PPO(
            policy=CTDEPolicy,
            env=self.vec_env,
            verbose=1,
            device=self.device,
            tensorboard_log="./tensorboard_logs/ctde_multi_agent",
            policy_kwargs={
                'features_extractor_class': CTDEFeatureExtractor,
                'features_extractor_kwargs': {
                    'global_observation_dim': global_observation_dim
                },
                'net_arch': dict(pi=[256, 256], vf=[256, 256])  # Keep larger architecture for capacity
            },
            learning_rate=2.6e-5,  # Slightly lower to stabilize late training
            n_steps=1024,  # Keep as is for better credit assignment
            batch_size=512,  # Keep as is for stable gradient updates
            gamma=0.98,  # Keep prioritizing short-term rewards
            gae_lambda=0.92,  # Keep slightly lower for stable advantage estimation
            clip_range=lambda progress_remaining: 0.2 * progress_remaining + 0.05,  # Reduce faster to stabilize late training
            ent_coef=0.03,  # Keep exploration incentive
            vf_coef=0.65,  # Keep reduced to prevent overfitting
            max_grad_norm=0.5,  # Keep to avoid instability
            **(model_kwargs if model_kwargs else {})
        )


        debug_print(f"PPO model initialized with updated architecture and hyperparameters.")  # Debugging

    def train(self, total_timesteps=1_000_000, checkpoint_freq=100_000, checkpoint_dir="./checkpoints_ctde_multi_agent/", checkpoint_prefix="ppo_ctde_multi_agent"):
        """
        Train the PPO model with centralized critic.
        
        Parameters:
        - total_timesteps: Total timesteps for training.
        - checkpoint_freq: Frequency (in timesteps) to save checkpoints.
        - checkpoint_dir: Directory to save checkpoints.
        - checkpoint_prefix: Prefix for checkpoint filenames.
        """
        # Define checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=checkpoint_prefix,
            verbose=1
        )
        debug_print(f"Checkpoint callback created with freq={checkpoint_freq}, path={checkpoint_dir}, prefix={checkpoint_prefix}")  # Debugging
        
        debug_print(f"Starting training for {total_timesteps} timesteps.")  # Debugging
        
        # Start training
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback]  # Only checkpoint_callback
            )
        except Exception as e:
            debug_print(f"Error during training: {e}")
            traceback.print_exc()
            raise e
        
        # Save the final model
        try:
            self.model.save("ppo_ctde_multi_agent_final")
            debug_print("Training completed and final model saved.")  # Debugging
        except Exception as e:
            print(f"Error during final model saving: {e}")
            traceback.print_exc()

    def evaluate(self, episodes=5, render=True):
        """
        Evaluate the trained PPO model with decentralized execution.
        
        Parameters:
        - episodes: Number of episodes to evaluate.
        - render: Whether to render the environment.
        """
        for episode in range(1, episodes + 1):
            try:
                obs, _ = self.vec_env.reset()
                debug_print(f"Resetting environment for evaluation episode {episode}.")
            except Exception as e:
                print(f"Error during environment reset: {e}")
                traceback.print_exc()
                break

            done = False
            step = 0
            total_reward = 0.0
            tasks_completed = 0
            
            print(f"\n=== Starting Evaluation Episode {episode} ===")

            while not done:
                # Predict action using the model
                try:
                    action, _states = self.model.predict(obs, deterministic=True)
                    debug_print(f"Predicted action: {action}")
                except Exception as e:
                    print(f"Error during action prediction: {e}")
                    traceback.print_exc()
                    break

                # Step the environment with the predicted action
                try:
                    obs, reward, terminated, truncated, info = self.vec_env.step(action)
                    done = terminated or truncated
                    debug_print(f"Step {step}: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Done={done}")
                except Exception as e:
                    print(f"Error during environment step: {e}")
                    traceback.print_exc()
                    break

                # Accumulate rewards
                total_reward += reward

                # Render the environment
                if render:
                    try:
                        self.vec_env.envs[0].render()
                    except Exception as e:
                        print(f"Error during rendering: {e}")
                        traceback.print_exc()
                        break

                step += 1
                tasks_completed = info.get("tasks_completed", 0)
            
            # Logging after each episode
            print(f"=== Evaluation Episode {episode} Finished ===")
            print(f"Total Steps: {step}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Tasks Completed: {tasks_completed} out of {self.env.num_tasks}")
            print("-" * 50)

    def save_normalizer(self, filepath="vec_normalize.pkl"):
        """
        Save the VecNormalize statistics.

        Parameters:
        - filepath: Path to save the normalization statistics.
        """
        if isinstance(self.vec_env, VecNormalize):
            self.vec_env.save(filepath)
            print(f"VecNormalize statistics saved to {filepath}")
        else:
            print("VecNormalize is not being used; nothing to save.")

    def load_normalizer(self, filepath="vec_normalize.pkl"):
        """
        Load the VecNormalize statistics.

        Parameters:
        - filepath: Path to load the normalization statistics from.
        """
        if isinstance(self.vec_env, VecNormalize):
            self.vec_env.load(filepath)
            print(f"VecNormalize statistics loaded from {filepath}")
        else:
            print("VecNormalize is not being used; nothing to load.")

# ==========================
# Function to Initialize Environment
# ==========================
def initialize_environment(env_config):
    """
    Initialize and validate the multi-agent environment.
    
    Parameters:
    - env_config: Dictionary containing environment configuration.
    
    Returns:
    - env: The initialized and validated environment.
    """
    print("Initializing Environment with config:", env_config)
    env = MultiRobotEnvSB3CollabTrue(**env_config)
    try:
        check_env(env, warn=True)
        print("Environment validation successful.")
    except AssertionError as e:
        print(f"Environment validation failed: {e}")
        traceback.print_exc()
        raise e
    except Exception as e:
        print(f"An error occurred during environment validation: {e}")
        traceback.print_exc()
        raise e
    return env

# ==========================
# Function to Load or Initialize Model
# ==========================
def load_or_initialize_model(env, checkpoint_dir, checkpoint_prefix, device='cpu', use_vec_normalize=True):
    """
    Load the latest checkpoint if available, else initialize a new PPO model.
    
    Parameters:
    - env: The multi-agent environment.
    - checkpoint_dir: Directory where checkpoints are saved.
    - checkpoint_prefix: Prefix of the checkpoint files.
    - device: 'cpu' or 'cuda'.
    - use_vec_normalize: Whether VecNormalize is used.
    
    Returns:
    - trainer: An instance of MultiAgentCTDETrainer with the loaded or new model.
    """
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        debug_print(f"Created checkpoint directory: {checkpoint_dir}")
    
    # Prompt the user
    while True:
        user_input = input("Do you want to continue training from the latest checkpoint? (y/n): ").strip().lower()
        if user_input in ['y', 'n']:
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
    
    trainer = MultiAgentCTDETrainer(env, device=device, use_vec_normalize=use_vec_normalize)
    
    if user_input == 'y':
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
        if latest_checkpoint:
            try:
                debug_print(f"Loading model from checkpoint: {latest_checkpoint}")  # Debugging
                trainer.model = PPO.load(latest_checkpoint, env=trainer.vec_env, verbose=1, device=device)
                if use_vec_normalize:
                    trainer.load_normalizer(os.path.join(checkpoint_dir, "vec_normalize.pkl"))
                print(f"Loaded model from checkpoint: {latest_checkpoint}")
                return trainer
            except Exception as e:
                print(f"Failed to load checkpoint '{latest_checkpoint}'. Starting a new training run.")
                traceback.print_exc()
        else:
            print("No checkpoints found. Starting a new training run.")
    
    # Initialize a new model
    print("Initializing a new PPO model.")
    return trainer

# ==========================
# Function to Run Training
# ==========================
def run_training(trainer, total_timesteps=10_000_000, checkpoint_freq=100_000, checkpoint_dir="./checkpoints_ctde_multi_agent/", checkpoint_prefix="ppo_ctde_multi_agent"):
    """
    Run the training process.
    
    Parameters:
    - trainer: An instance of MultiAgentCTDETrainer.
    - total_timesteps: Total timesteps for training.
    - checkpoint_freq: Frequency (in timesteps) to save checkpoints.
    - checkpoint_dir: Directory where checkpoints are saved.
    - checkpoint_prefix: Prefix for checkpoint filenames.
    """
    trainer.train(total_timesteps=total_timesteps, checkpoint_freq=checkpoint_freq, checkpoint_dir=checkpoint_dir, checkpoint_prefix=checkpoint_prefix)

# ==========================
# Function to Run Evaluation
# ==========================
def run_evaluation(trainer, episodes=5, render=True):
    """
    Run the evaluation process.
    
    Parameters:
    - trainer: An instance of MultiAgentCTDETrainer.
    - episodes: Number of episodes to evaluate.
    - render: Whether to render the environment.
    """
    trainer.evaluate(episodes=episodes, render=render)

# ==========================
# Main Execution
# ==========================

def main():
    # Define environment configuration
    env_config = {
        "num_robots": 4,
        "num_tasks": 6,
        "field_size": 15.0,
        "max_episode_steps": 300,
        "collision_penalty": True,       # Penalize collisions
        "completion_radius": 0.5,
        "robot_radius": 0.3,
        "time_penalty": -1.0,            # Updated per timestep penalty
        "max_velocity": 1.0,
        "max_angular_velocity": 1.0,
        "task_weights": None,            # Can be set to a list of weights or left as None for random assignment
    }

    # Initialize and validate the environment
    try:
        env = initialize_environment(env_config)
    except Exception:
        print("Failed to initialize the environment. Exiting.")
        return

    # Define checkpoint parameters
    checkpoint_dir = "./checkpoints_ctde_multi_agent/"
    checkpoint_prefix = "ppo_ctde_multi_agent"

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Using GPU for training.")
    else:
        print("Using CPU for training.")

    # Load or initialize the PPO model
    trainer = load_or_initialize_model(env, checkpoint_dir, checkpoint_prefix, device=device, use_vec_normalize=True)

    # Start training
    try:
        print("Starting training...")
        run_training(trainer, total_timesteps=10_000_000, checkpoint_freq=100_000, checkpoint_dir=checkpoint_dir, checkpoint_prefix=checkpoint_prefix)
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return
    print("Training completed and model saved.")

    # Save VecNormalize statistics
    try:
        trainer.save_normalizer(os.path.join(checkpoint_dir, "vec_normalize.pkl"))
    except Exception as e:
        print(f"Error during VecNormalize saving: {e}")
        traceback.print_exc()

    # Evaluation parameters
    num_evaluation_episodes = 5
    render_env = True  # Set to False to disable rendering

    # Run evaluation
    try:
        run_evaluation(trainer, episodes=num_evaluation_episodes, render=render_env)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

    # Close the environment
    try:
        trainer.vec_env.close()
        print("Environment closed.")
    except Exception as e:
        print(f"Error during environment closure: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
