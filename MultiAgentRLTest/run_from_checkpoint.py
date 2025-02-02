# visualize_checkpoint.py

import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from MultiAgent.MultiAgentRLTest.multi_robot_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize Multi-Agent PPO Checkpoints")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the PPO checkpoint file (e.g., ./checkpoints/ppo_ctde_multi_agent_100000_steps.zip)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to visualize (default: 5)'
    )
    parser.add_argument(
        '--render_delay',
        type=float,
        default=0.05,
        help='Delay between frames in seconds for rendering (default: 0.05)'
    )
    parser.add_argument(
        '--disable_render',
        action='store_true',
        help='Disable rendering (useful for testing without visualization)'
    )
    return parser.parse_args()

def initialize_environment():
    """
    Initializes the Multi-Agent Environment with predefined configuration.

    Returns:
        env: Initialized MultiRobotEnvSB3CollabTrue environment.
    """
    env_config = {
        "num_robots": 4,
        "num_tasks": 6,
        # "robots_per_task": 2,  # Removed as it's no longer used
        "field_size": 15.0,
        "max_episode_steps": 300,
        "collision_penalty": True,  # Penalize collisions
        "completion_radius": 0.5,
        "robot_radius": 0.3,
        "time_penalty": -0.1,
        "max_velocity": 1.0,
        "max_angular_velocity": 1.0,
        "task_weights": None,  # Can be set to a list of weights or left as None for random assignment
    }

    env = MultiRobotEnvSB3CollabTrue(**env_config)
    return env

def load_model(checkpoint_path, env, device='cpu'):
    """
    Loads the PPO model from the specified checkpoint.

    Args:
        checkpoint_path (str): Path to the PPO checkpoint file.
        env: The environment to associate with the model.
        device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        model: Loaded PPO model.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = PPO.load(checkpoint_path, env=env, device=device)
    print(f"Loaded model from checkpoint: {checkpoint_path} on device: {device}")
    return model

def visualize(model, env, num_episodes=5, render_delay=0.05, disable_render=False):
    """
    Runs episodes using the loaded model and visualizes the agents' actions.

    Args:
        model: Loaded PPO model.
        env: MultiRobotEnvSB3CollabTrue environment.
        num_episodes (int): Number of episodes to visualize.
        render_delay (float): Delay between frames in seconds for rendering.
        disable_render (bool): If True, disables rendering.
    """
    for episode in range(1, num_episodes + 1):
        try:
            observation, info = env.reset()
            done = False
            step = 0
            total_reward = 0.0
            tasks_completed = 0

            print(f"\n=== Starting Episode {episode} ===")

            while not done:
                if not disable_render:
                    env.render()

                # Predict action using the model
                action, _states = model.predict(observation, deterministic=True)

                # Take a step in the environment
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Accumulate rewards and tasks completed
                total_reward += reward
                tasks_completed = info.get("tasks_completed", 0)

                step += 1

                # Optional: Add a short delay for better visualization
                if not disable_render:
                    time.sleep(render_delay)

            # Print episode summary
            print(f"=== Episode {episode} Finished ===")
            print(f"Total Steps: {step}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Tasks Completed: {tasks_completed} out of {env.num_tasks}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nVisualization interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred during Episode {episode}: {e}")
            env.close()
            raise e  # Re-raise the exception after cleanup

    # Close the environment after all episodes
    env.close()
    print("\nAll episodes completed and environment closed.")

def main():
    args = parse_arguments()

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize environment
    env = initialize_environment()

    # Load model
    model = load_model(args.checkpoint, env, device=device)

    # Visualize episodes
    visualize(
        model=model,
        env=env,
        num_episodes=args.episodes,
        render_delay=args.render_delay,
        disable_render=args.disable_render
    )

if __name__ == "__main__":
    main()
