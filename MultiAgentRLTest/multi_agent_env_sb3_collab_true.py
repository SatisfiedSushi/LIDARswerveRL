# multi_agent_env_sb3_collab_true.py

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MultiRobotEnvSB3CollabTrue(gym.Env):
    """
    A Gym-compatible environment for true multi-robot collaboration.
    Each robot acts independently with its own policy.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        num_robots=2,
        num_tasks=3,
        robots_per_task=2,
        field_size=10.0,
        max_episode_steps=200,
        collision_penalty=False,
        completion_radius=0.5,
        robot_radius=0.3,
        time_penalty=-0.1,
        max_velocity=1.0,
        max_angular_velocity=1.0,
    ):
        super(MultiRobotEnvSB3CollabTrue, self).__init__()
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.robots_per_task = robots_per_task
        self.field_size = field_size
        self.max_episode_steps = max_episode_steps
        self.collision_penalty = collision_penalty
        self.completion_radius = completion_radius
        self.robot_radius = robot_radius
        self.time_penalty = time_penalty
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity

        # Define action and observation spaces for each agent
        # Actions: [velocity_x, velocity_y, angular_velocity]
        self.action_spaces = {
            f"robot_{i}": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for i in range(self.num_robots)
        }

        # Observations: [position_x, position_y, orientation, velocity_x, velocity_y, angular_velocity] + task info
        single_obs_dim = 6 + 3 * self.num_tasks
        self.observation_spaces = {
            f"robot_{i}": spaces.Box(low=-np.inf, high=np.inf, shape=(single_obs_dim,), dtype=np.float32)
            for i in range(self.num_robots)
        }

        # Initialize state variables
        self.reset()

        # Initialize visualization components
        self.viewer = None
        self.fig, self.ax = None, None
        self.robot_patches = []
        self.orientation_arrows = []
        self.active_tasks_patches = []
        self.inactive_tasks_patches = []

    def reset(self):
        self.steps = 0
        # Initialize robot states
        self.robot_positions = np.random.uniform(0, self.field_size, (self.num_robots, 2)).astype(np.float32)
        self.robot_orientations = np.random.uniform(-np.pi, np.pi, self.num_robots).astype(np.float32)
        self.robot_velocities = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_angular_vel = np.zeros(self.num_robots, dtype=np.float32)

        # Initialize task states
        self.task_positions = np.random.uniform(0, self.field_size, (self.num_tasks, 2)).astype(np.float32)
        self.task_active = np.ones(self.num_tasks, dtype=bool)
        self.task_assignments = {t: [] for t in range(self.num_tasks)}  # Track assigned robots

        # Return observations for each agent
        observations = {
            f"robot_{i}": self._get_obs(i)
            for i in range(self.num_robots)
        }

        return observations

    def step(self, actions):
        self.steps += 1
        rewards = {f"robot_{i}": 0.0 for i in range(self.num_robots)}
        dones = {f"robot_{i}": False for i in range(self.num_robots)}
        infos = {f"robot_{i}": {} for i in range(self.num_robots)}

        # Update robot velocities based on actions
        for i, agent in enumerate(actions.keys()):
            action = actions[agent]
            desired_vx = np.clip(action[0], -1.0, 1.0) * self.max_velocity
            desired_vy = np.clip(action[1], -1.0, 1.0) * self.max_velocity
            desired_w  = np.clip(action[2], -1.0, 1.0)  * self.max_angular_velocity

            self.robot_velocities[i] = [desired_vx, desired_vy]
            self.robot_angular_vel[i] = desired_w

        # Integrate positions and orientations
        dt = 0.1  # Time step
        for i in range(self.num_robots):
            self.robot_positions[i] += self.robot_velocities[i] * dt
            self.robot_positions[i] = np.clip(self.robot_positions[i], 0, self.field_size)
            self.robot_orientations[i] += self.robot_angular_vel[i] * dt

        # Reset task assignments
        self.task_assignments = {t: [] for t in range(self.num_tasks)}

        # Assign robots to tasks based on proximity
        for i in range(self.num_robots):
            for t in range(self.num_tasks):
                if self.task_active[t]:
                    tx, ty = self.task_positions[t]
                    rx, ry = self.robot_positions[i]
                    dist = np.linalg.norm([rx - tx, ry - ty])
                    if dist < self.completion_radius:
                        self.task_assignments[t].append(i)

        # Compute rewards and handle task completions
        for t in range(self.num_tasks):
            if self.task_active[t] and len(self.task_assignments[t]) >= self.robots_per_task:
                # Reward all robots assigned to this task
                for robot_id in self.task_assignments[t]:
                    rewards[f"robot_{robot_id}"] += 10.0
                self.task_active[t] = False

        # Apply time penalty
        for agent in rewards.keys():
            rewards[agent] += self.time_penalty

        # Check termination conditions
        all_tasks_done = not np.any(self.task_active)
        done = self.steps >= self.max_episode_steps or all_tasks_done
        dones = {agent: done for agent in dones.keys()}
        dones["__all__"] = done

        # Generate observations
        observations = {
            f"robot_{i}": self._get_obs(i)
            for i in range(self.num_robots)
        }

        return observations, rewards, dones, infos

    def _get_obs(self, agent_i):
        """
        Returns the observation for a specific agent.
        """
        obs = []
        # Robot's own state
        obs.extend([
            float(self.robot_positions[agent_i, 0]),
            float(self.robot_positions[agent_i, 1]),
            float(self.robot_orientations[agent_i]),
            float(self.robot_velocities[agent_i, 0]),
            float(self.robot_velocities[agent_i, 1]),
            float(self.robot_angular_vel[agent_i]),
        ])
        # Task states
        for t in range(self.num_tasks):
            obs.extend([
                float(self.task_positions[t, 0]),
                float(self.task_positions[t, 1]),
                1.0 if self.task_active[t] else 0.0
            ])
        return np.array(obs, dtype=np.float32)

    def render(self, mode='human'):
        """
        Renders the current state of the environment using Matplotlib.
        """
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8,8))
            self.ax.set_xlim(0, self.field_size)
            self.ax.set_ylim(0, self.field_size)
            self.ax.set_aspect('equal')
            self.ax.set_title("True Multi-Robot Collaborative Environment")

            # Initialize robot patches and orientation arrows
            self.robot_patches = []
            self.orientation_arrows = []
            for i in range(self.num_robots):
                robot = patches.Circle((0,0), self.robot_radius, fc='blue', ec='black', alpha=0.6)
                self.robot_patches.append(robot)
                self.ax.add_patch(robot)
                # Arrow for orientation
                arrow = self.ax.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
                self.orientation_arrows.append(arrow)

            # Initialize task patches
            self.active_tasks_patches = []
            self.inactive_tasks_patches = []
            for t in range(self.num_tasks):
                task = patches.Circle((0,0), 0.2, fc='green', ec='black', alpha=0.6)
                self.active_tasks_patches.append(task)
                self.ax.add_patch(task)
            self.fig.canvas.draw()

            # Initialize text elements
            self.reward_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
            self.steps_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes)
            self.tasks_text = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes)

        # Update robot positions and orientations
        for i in range(self.num_robots):
            pos = self.robot_positions[i]
            ori = self.robot_orientations[i]
            self.robot_patches[i].center = (pos[0], pos[1])

            # Remove old arrow
            self.orientation_arrows[i].remove()
            # Calculate new arrow direction
            arrow_length = self.robot_radius
            dx = arrow_length * np.cos(ori)
            dy = arrow_length * np.sin(ori)
            arrow = self.ax.arrow(pos[0], pos[1], dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
            self.orientation_arrows[i] = arrow

        # Update tasks
        # First, remove all task patches
        for task in self.active_tasks_patches + self.inactive_tasks_patches:
            task.remove()
        self.active_tasks_patches = []
        self.inactive_tasks_patches = []
        for t in range(self.num_tasks):
            tx, ty = self.task_positions[t]
            if self.task_active[t]:
                task = patches.Circle((tx, ty), 0.2, fc='green', ec='black', alpha=0.6)
                self.active_tasks_patches.append(task)
                self.ax.add_patch(task)
            else:
                task_inactive = patches.Circle((tx, ty), 0.2, fc='gray', ec='black', alpha=0.3)
                self.inactive_tasks_patches.append(task_inactive)
                self.ax.add_patch(task_inactive)

        # Update text elements
        # For visualization purposes, assume a global reward and step counter
        # Modify as needed to pass these from the main loop
        # Here, they are placeholders
        # self.reward_text.set_text(f'Total Reward: {total_reward:.2f}')
        # self.steps_text.set_text(f'Steps: {self.steps}')
        # self.tasks_text.set_text(f'Tasks Completed: {info.get("tasks_completed", 0)}/{self.num_tasks}')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig, self.ax = None, None
