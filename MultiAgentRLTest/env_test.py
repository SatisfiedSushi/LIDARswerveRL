# visualize_multi_robot_env.py

import gymnasium as gym
from multi_robot_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue
import numpy as np
import time

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
        "normalize_observations": False, # Disable normalization for visualization
        "normalize_rewards": False,      # Disable normalization for visualization
    }

    # ==========================
    # Initialize the Environment
    # ==========================
    env = MultiRobotEnvSB3CollabTrue(**env_config)

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
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        tasks_completed = 0

        print(f"\n=== Starting Episode {episode} ===")

        while not done:
            # ==========================
            # Sample Random Actions
            # ==========================
            action = env.action_space.sample()  # Sample random actions for all robots

            # ==========================
            # Step the Environment
            # ==========================
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # ==========================
            # Render the Environment
            # ==========================
            if render:
                env.render()

            # ==========================
            # Optional: Print Step Information
            # ==========================
            # Uncomment the following line to see step-by-step details
            # print(f"Step {step}: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Done={done}")

            step += 1
            tasks_completed = info.get("tasks_completed", 0)

            # ==========================
            # Pause for Visualization
            # ==========================
            time.sleep(sleep_time)

        # ==========================
        # Episode Summary
        # ==========================
        print(f"=== Episode {episode} Finished ===")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Tasks Completed: {tasks_completed} out of {env.num_tasks}")
        print("-" * 50)

    # ==========================
    # Close the Environment
    # ==========================
    env.close()
    print("Visualization completed and environment closed.")

if __name__ == "__main__":
    main()
