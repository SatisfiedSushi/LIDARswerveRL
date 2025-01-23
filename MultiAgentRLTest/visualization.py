# visualize_random_multi_robot_env.py

import time
from multi_agent_env_sb3_collab_true import MultiRobotEnvSB3CollabTrue  # Ensure correct import path

def main():
    # Define environment configuration
    env_config = {
        "num_robots": 4,
        "num_tasks": 6,
        "robots_per_task": 2,  # Collaborative tasks
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

    try:
        while True:
            obs = env.reset()
            done = False
            step = 0
            total_reward = 0.0

            while not done:
                # Sample random actions for all robots
                actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in env.action_spaces.keys()}
                obs, rewards, dones, infos = env.step(actions)
                total_reward += sum(rewards.values())
                step += 1

                # Render the current state of the environment
                env.render()

                # Optional: Add a short delay for better visualization
                time.sleep(0.05)

            print(f"Episode finished after {step} steps with total reward {total_reward}")
            print(f"Tasks Completed: {infos.get('tasks_completed', 0)} out of {env.num_tasks}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("Visualization interrupted by user.")

    finally:
        # Close the environment and Matplotlib windows
        env.close()

if __name__ == "__main__":
    main()
