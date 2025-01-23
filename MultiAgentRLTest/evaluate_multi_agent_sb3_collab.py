# evaluate_multi_agent_sb3_collab.py

import gym
import numpy as np
from stable_baselines3 import PPO
from MultiAgent.MultiAgentRLTest.multi_agent_env_sb3_collab_true import MultiRobotEnvSB3Collab  # Ensure correct import path

if __name__ == "__main__":
    # Load the trained model
    model = PPO.load("ppo_multi_robot_collab_final")

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
    env = MultiRobotEnvSB3Collab(**env_config)

    # Reset the environment
    obs = env.reset()

    total_reward = 0.0
    done = False
    step = 0

    while not done:
        # Predict actions
        action, _states = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, reward, done, info = env.step(action)

        total_reward += reward
        step += 1

        # Optionally, render the environment
        # env.render()

    print(f"Episode finished after {step} steps with total reward {total_reward}")
    print(f"Tasks Completed: {info.get('tasks_completed', 0)} out of {env.num_tasks}")
