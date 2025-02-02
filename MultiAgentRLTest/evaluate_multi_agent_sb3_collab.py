# visualize_trained_model.py

import gymnasium as gym
from multi_robot_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue
import numpy as np
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import traceback

def main():
    # ==========================
    # Environment Configuration
    # ==========================
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
        "normalize_observations": True,  # Enable normalization to match training
        "normalize_rewards": True,       # Enable normalization to match training
    }

    # ==========================
    # Initialize the Environment
    # ==========================
    env = MultiRobotEnvSB3CollabTrue(**env_config)

    # ==========================
    # Wrap the Environment
    # ==========================
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load("./checkpoints_ctde_multi_agent/vec_normalize.pkl", vec_env)
    vec_env.training = False  # Disable training for VecNormalize
    vec_env.norm_reward = False  # Do not normalize rewards during evaluation

    # ==========================
    # Load the Trained Model
    # ==========================
    model_path = "ppo_ctde_multi_agent_final.zip"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {model_path} on device {device}...")
    try:
        model = PPO.load(model_path, env=vec_env, device=device)
    except Exception as e:
        print(f"Error loading the model: {e}")
        traceback.print_exc()
        return

    # ==========================
    # Visualization Parameters
    # ==========================
    num_episodes = 3          # Number of episodes to visualize
    render = True             # Whether to render the environment
    sleep_time = 0.05         # Time (in seconds) to wait between steps for visualization

    # ==========================
    # Run Visualization Episodes
    # ==========================
    for episode in range(1, num_episodes + 1):
        try:
            obs = vec_env.reset()
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
            # ==========================
            # Predict Action Using the Trained Model
            # ==========================
            try:
                action, _states = model.predict(obs, deterministic=True)
            except Exception as e:
                print(f"Error during action prediction: {e}")
                traceback.print_exc()
                break

            # ==========================
            # Step the Environment with the Predicted Action
            # ==========================
            try:
                obs, reward, done, info = vec_env.step(action)
            except Exception as e:
                print(f"Error during environment step: {e}")
                traceback.print_exc()
                break

            # ==========================
            # Accumulate Rewards
            # ==========================
            total_reward += reward

            # ==========================
            # Render the Environment
            # ==========================
            if render:
                try:
                    vec_env.envs[0].render()
                except Exception as e:
                    print(f"Error during rendering: {e}")
                    traceback.print_exc()
                    break

            # ==========================
            # Optional: Print Step Information
            # ==========================
            # Uncomment the following line to see step-by-step details
            # print(f"Step {step}: Reward={reward}, Done={done}")
            
            step += 1
            if isinstance(info, list) and len(info) > 0:
                tasks_completed = info[0].get("tasks_completed", 0)
            else:
                tasks_completed = 0

            # ==========================
            # Pause for Visualization
            # ==========================
            time.sleep(sleep_time)

        # ==========================
        # Episode Summary
        # ==========================
        print(f"=== Evaluation Episode {episode} Finished ===")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Tasks Completed: {tasks_completed} out of {env.num_tasks}")
        print("-" * 50)

    # ==========================
    # Close the Environment
    # ==========================
    vec_env.close()
    print("Visualization completed and environment closed.")

if __name__ == "__main__":
    main()
