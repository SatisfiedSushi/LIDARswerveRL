# infer_with_tensorrt.py

import torch
from multi_agent_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue  # Ensure correct import path
import matplotlib.pyplot as plt
import numpy as np

class TensorRTRunner:
    def __init__(self, trt_model_path, agent_id, device='cuda'):
        """
        Initializes the TensorRTRunner.

        Parameters:
        - trt_model_path: Path to the TensorRT optimized model.
        - agent_id: Identifier of the agent.
        - device: 'cuda' for GPU inference.
        """
        self.trt_model = torch.jit.load(trt_model_path).to(device)
        self.trt_model.eval()
        self.agent_id = agent_id
        self.device = device

    def predict(self, observation):
        """
        Runs inference on the observation using the TensorRT model.

        Parameters:
        - observation: NumPy array representing the agent's observation.

        Returns:
        - action: Predicted action as a NumPy array.
        """
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.trt_model(obs_tensor)
        return action.cpu().numpy().flatten()

def visualize_inference(env, runners, episodes=1, render=True):
    """
    Runs inference with TensorRT models and visualizes the environment.

    Parameters:
    - env: The multi-agent environment.
    - runners: Dictionary mapping agent IDs to TensorRTRunner instances.
    - episodes: Number of episodes to run.
    - render: Whether to render the environment.
    """
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        step = 0
        total_rewards = {agent_id: 0.0 for agent_id in runners.keys()}

        while not done:
            actions = {}
            for agent_id, runner in runners.items():
                # Get observation for the agent
                agent_obs = obs[agent_id]
                # Predict action using TensorRT model
                action = runner.predict(agent_obs)
                # Clip actions to valid range
                action = np.clip(action, -1.0, 1.0)
                actions[agent_id] = action

            # Take a step in the environment
            obs, rewards, dones, infos = env.step(actions)

            # Accumulate rewards
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward

            if render:
                env.render()

            done = dones["__all__"]
            step += 1

        print(f"Episode {episode} finished after {step} steps.")
        for agent_id, reward in total_rewards.items():
            print(f"  {agent_id} Total Reward: {reward:.2f}")
        print(f"  Tasks Completed: {infos.get('tasks_completed', 0)} out of {env.num_tasks}")
        print("-" * 50)

if __name__ == "__main__":
    # Define environment configuration
    env_config = {
        "num_robots": 4,
        "num_tasks": 6,
        "robots_per_task": 2,
        "field_size": 15.0,
        "max_episode_steps": 300,
        "collision_penalty": False,
        "completion_radius": 0.5,
        "robot_radius": 0.3,
        "time_penalty": -0.1,
        "max_velocity": 1.0,
        "max_angular_velocity": 1.0,
    }

    # Create the environment
    env = MultiRobotEnvSB3CollabTrue(**env_config)

    # Initialize TensorRT runners for each agent
    trt_model_paths = {
        "robot_0": "./trt_models/ppo_multi_agent_robot_0_trt.pt",
        "robot_1": "./trt_models/ppo_multi_agent_robot_1_trt.pt",
        "robot_2": "./trt_models/ppo_multi_agent_robot_2_trt.pt",
        "robot_3": "./trt_models/ppo_multi_agent_robot_3_trt.pt",
    }

    runners = {}
    for agent_id, trt_path in trt_model_paths.items():
        runners[agent_id] = TensorRTRunner(trt_path, agent_id, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Run inference and visualize
    visualize_inference(env, runners, episodes=3, render=True)

    # Close the environment
    env.close()
