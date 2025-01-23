# train_multi_agent_ctde.py

import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from multi_agent_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue  # Ensure correct import path
import gym

# ==========================
# Custom Policy for CTDE
# ==========================

class CTDEPolicy(ActorCriticPolicy):
    """
    Custom Policy class implementing Centralized Training with Decentralized Execution (CTDE).
    
    In this setup:
    - **Actor (Policy Network):** Each agent has its own policy network that takes only its local observations.
    - **Critic (Value Network):** A centralized critic that takes the global state (joint observations) to estimate the value function.
    
    This policy ensures that during training, the critic has access to the global state for better coordination,
    while during execution, each agent relies solely on its local observations.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CTDEPolicy.
        """
        super(CTDEPolicy, self).__init__(*args, **kwargs)
        
        # Define the critic network to take the global state as input
        # Assuming that the environment provides the global state as a separate observation component
        # Modify the input dimension accordingly
        global_observation_dim = kwargs['features_extractor_kwargs']['global_observation_dim']
        self.critic_network = nn.Sequential(
            nn.Linear(global_observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, obs, deterministic=False):
        """
        Forward pass for the actor (policy network).
        """
        # 'obs' contains only the local observation for the agent
        return self._build_action(obs, deterministic=deterministic)
    
    def _predict_values(self, obs, actions=None):
        """
        Predict value estimates using the centralized critic.
        
        Parameters:
        - obs: Dictionary containing local observations and global state.
        - actions: Actions taken by agents (optional).
        
        Returns:
        - value: Value estimates from the centralized critic.
        """
        # Extract the global state from the observation
        global_state = obs['global_state']
        global_state = torch.as_tensor(global_state).float().to(self.device)
        
        # Pass the global state through the critic network
        value = self.critic_network(global_state)
        return value

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions by computing the log probabilities and entropy.
        Also compute the value estimates using the centralized critic.
        
        Parameters:
        - obs: Dictionary containing local observations and global state.
        - actions: Actions taken by agents.
        
        Returns:
        - values: Value estimates from the centralized critic.
        - log_prob: Log probabilities of the actions.
        - entropy: Entropy of the action distribution.
        """
        # Forward pass through the actor to get distribution
        distribution = self._get_action_dist_from_latent(self._build_latent(obs))
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Compute the value using the centralized critic
        values = self._predict_values(obs, actions)
        
        return values, log_prob, entropy

# ==========================
# Custom Feature Extractor
# ==========================

class CTDEFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor that separates local and global observations.
    
    Assumes that the observation is a dictionary with 'local' and 'global' keys.
    """

    def __init__(self, observation_space: gym.spaces.Dict, global_observation_dim: int):
        """
        Initialize the feature extractor.
        
        Parameters:
        - observation_space: The observation space of the environment.
        - global_observation_dim: Dimension of the global observation.
        """
        super(CTDEFeatureExtractor, self).__init__(observation_space, features_dim=256)
        
        # Extract the local observation space
        self.local_observation_space = observation_space.spaces['local']
        
        # Define the feature extractor for the local observation
        self.local_net = nn.Sequential(
            nn.Linear(self.local_observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Define the feature extractor for the global observation
        self.global_net = nn.Sequential(
            nn.Linear(global_observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
    def forward(self, observations):
        """
        Forward pass to extract features from observations.
        
        Parameters:
        - observations: A dictionary containing 'local' and 'global' observations.
        
        Returns:
        - features: Concatenated features from local and global observations.
        """
        local_obs = observations['local']
        global_obs = observations['global']
        
        local_features = self.local_net(local_obs)
        global_features = self.global_net(global_obs)
        
        # Concatenate local and global features
        features = torch.cat((local_features, global_features), dim=1)
        return features

# ==========================
# Multi-Agent Trainer with CTDE
# ==========================

class MultiAgentCTDETrainer:
    """
    Trainer class implementing Centralized Training with Decentralized Execution (CTDE).
    
    Uses a shared PPO model with a centralized critic for homogeneous agents.
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
        # Assuming the environment provides a separate 'global_state' observation
        # Modify accordingly based on your environment's actual observation structure
        sample_obs = env.reset()
        if 'global_state' in sample_obs:
            global_observation = sample_obs['global_state']
            global_observation_dim = global_observation.shape[0]
        else:
            raise ValueError("Environment must provide 'global_state' in observations for centralized critic.")
        
        # Define the observation space as a Dict with 'local' and 'global'
        # This requires modifying the environment to provide observations in this format
        # 'local' is a list of local observations for each agent
        # 'global' is the global state
        # For training, we need to reshape the observations accordingly
        
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
                }
            },
            **(model_kwargs if model_kwargs else {})
        )
    
    def prepare_observations(self, observations):
        """
        Preprocess observations to include both local and global states.
        
        Parameters:
        - observations: Raw observations from the environment (dict).
        
        Returns:
        - processed_obs: Dictionary with 'local' and 'global' keys.
        """
        # Extract global state
        global_state = self.env.get_global_state()  # Implement this method in your environment
        # Extract local observations and stack them
        local_observations = []
        for agent_id in sorted(observations.keys()):
            local_observations.append(observations[agent_id])
        local_observations = np.stack(local_observations, axis=0)  # Shape: (num_agents, obs_dim)
        
        # Prepare the processed observation
        processed_obs = {
            'local': torch.from_numpy(local_observations).float(),
            'global': torch.from_numpy(global_state).float()
        }
        return processed_obs
    
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
        
        # Start training
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback
        )
        
        # Save the final model
        self.model.save("ppo_ctde_multi_agent_final")
        print("Training completed and model saved.")
    
    def evaluate(self, episodes=5, render=True):
        """
        Evaluate the trained PPO model with decentralized execution.
        
        Parameters:
        - episodes: Number of episodes to evaluate.
        - render: Whether to render the environment.
        """
        for episode in range(1, episodes + 1):
            observations = self.env.reset()
            done = False
            step = 0
            total_rewards = {agent_id: 0.0 for agent_id in observations.keys()}
            tasks_completed = 0
            
            while not done:
                # Prepare the observation in the required format
                processed_obs = {
                    'local': torch.from_numpy(np.array([observations[agent] for agent in sorted(observations.keys())])).float(),
                    'global': torch.from_numpy(self.env.get_global_state()).float()
                }
                
                # Get actions from the model
                # Assuming the model outputs actions for all agents jointly
                actions, _states = self.model.predict(processed_obs, deterministic=True)
                
                # Split the actions for each agent
                # Assuming actions are concatenated for all agents
                num_agents = self.env.num_robots
                action_dim = self.env.action_spaces['robot_0'].shape[0]
                actions_dict = {}
                for i, agent_id in enumerate(sorted(observations.keys())):
                    actions_dict[agent_id] = actions[i * action_dim : (i + 1) * action_dim]
                
                # Step the environment
                observations, rewards, dones, infos = self.env.step(actions_dict)
                
                # Accumulate rewards
                for agent_id, reward in rewards.items():
                    total_rewards[agent_id] += reward
                
                if render:
                    self.env.render()
                
                done = dones["__all__"]
                step += 1
                tasks_completed = infos.get("tasks_completed", 0)
            
            # Logging after each episode
            print(f"Episode {episode} finished after {step} steps.")
            for agent_id, reward in total_rewards.items():
                print(f"  {agent_id} Total Reward: {reward:.2f}")
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
    
    # Ensure the environment has a method to get the global state
    # You need to implement this method in your environment
    # Example:
    # def get_global_state(self):
    #     # Concatenate all robots' positions, orientations, velocities, and task states
    #     global_state = np.concatenate([
    #         self.robot_positions.flatten(),
    #         self.robot_orientations,
    #         self.robot_velocities.flatten(),
    #         self.task_positions.flatten(),
    #         self.task_active.astype(float)
    #     ])
    #     return global_state
    if not hasattr(env, 'get_global_state'):
        raise AttributeError("The environment must implement a 'get_global_state' method for CTDE.")

    # Initialize the MultiAgentCTDETrainer
    trainer = MultiAgentCTDETrainer(
        env=env,
        model_kwargs={
            "learning_rate": 1e-4,
            "n_steps": 1024,
            "batch_size": 256,
            "n_epochs": 15,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.15,
            "ent_coef": 0.005, 
        },
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Start training
    trainer.train(
        total_timesteps=10_000_000,
        checkpoint_freq=100_000      # Save checkpoint every n steps
    )
    
    # Evaluate the trained model
    trainer.evaluate(
        episodes=5,
        render=True
    )
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
