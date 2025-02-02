# multi_robot_env_sb3_collab_true.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================
# Debug Flag and Utility Function
# ==========================
DEBUG = False  # Set to True to enable debug prints

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class MultiRobotEnvSB3CollabTrue(gym.Env):
    """
    A Gymnasium-compatible environment for true multi-robot collaboration.
    Implements Centralized Training with Decentralized Execution (CTDE).

    Collaborative Transport Task:
    - Robots collaboratively transport objects to target locations.
    - Each task has a weight determining the number of robots required to complete it.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        num_robots=4,
        num_tasks=6,
        field_size=15.0,
        max_episode_steps=300,
        collision_penalty=True,
        completion_radius=0.5,
        robot_radius=0.3,
        time_penalty=-1.0,  # Updated per timestep penalty
        max_velocity=1.0,
        max_angular_velocity=1.0,
        task_weights=None,  # List or array of weights per task
        normalize_observations=True,  # Enable observation normalization
        normalize_rewards=True,        # Enable reward normalization
    ):
        super(MultiRobotEnvSB3CollabTrue, self).__init__()
        debug_print("Initializing MultiRobotEnvSB3CollabTrue Environment")
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.field_size = field_size
        self.max_episode_steps = max_episode_steps
        self.collision_penalty = collision_penalty
        self.completion_radius = completion_radius
        self.robot_radius = robot_radius
        self.time_penalty = time_penalty
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.task_weights = task_weights if task_weights is not None else np.random.randint(2, 4, size=self.num_tasks)
        debug_print(f"Task Weights: {self.task_weights}")

        # Define a single, flattened action space for all agents
        # Each agent has 3 actions: [velocity_x, velocity_y, angular_velocity]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_robots * 3,),
            dtype=np.float32
        )
        debug_print(f"Action Space: {self.action_space}")

        # Enhanced observation space to include agent-to-task distances and relative positions
        # Observation components:
        # - Agent positions (x, y)
        # - Agent orientations
        # - Agent velocities (vx, vy)
        # - Agent angular velocities
        # - Task positions (object_x, object_y)
        # - Task target positions (target_x, target_y)
        # - Task active flags
        # - Task required robots
        # - Agent-to-task distances
        # - Agent-to-target distances

        observation_dim = (
            self.num_robots * 2 +    # robot_positions (x, y) for each robot
            self.num_robots +        # robot_orientations for each robot
            self.num_robots * 2 +    # robot_velocities (vx, vy) for each robot
            self.num_robots +        # robot_angular_vel for each robot
            self.num_tasks * 6 +     # For each task: object_x, object_y, target_x, target_y, active_flag, required_robots
            self.num_robots * self.num_tasks * 2  # Agent-to-task distances and Agent-to-target distances
        )
        debug_print(f"Global Observation Dimension: {observation_dim}")

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )
        debug_print(f"Observation Space: {self.observation_space}")

        # Normalization parameters
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        if self.normalize_observations:
            self.obs_mean = np.zeros(observation_dim, dtype=np.float32)
            self.obs_std = np.ones(observation_dim, dtype=np.float32)

        # Initialize state variables
        self.reset()

        # Initialize visualization components
        self.fig, self.ax = None, None
        self.robot_patches = []
        self.orientation_arrows = []
        self.object_patches = []
        self.target_patches = []
        self.active_tasks_patches = []
        self.inactive_tasks_patches = []
        self.text_elements = {}

    def reset(self, seed=None, options=None):
        debug_print("Resetting Environment")
        if seed is not None:
            np.random.seed(seed)
        
        self.steps = 0
        self.total_reward = 0.0
        self.tasks_completed = 0

        # Initialize robot states
        self.robot_positions = np.random.uniform(0, self.field_size, (self.num_robots, 2)).astype(np.float32)
        self.robot_orientations = np.random.uniform(-np.pi, np.pi, self.num_robots).astype(np.float32)
        self.robot_velocities = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_angular_vel = np.zeros(self.num_robots, dtype=np.float32)
        debug_print(f"Initial Robot Positions:\n{self.robot_positions}")
        debug_print(f"Initial Robot Orientations:\n{self.robot_orientations}")

        # Initialize task states
        self.object_positions = np.random.uniform(0, self.field_size, (self.num_tasks, 2)).astype(np.float32)
        self.target_positions = np.random.uniform(0, self.field_size, (self.num_tasks, 2)).astype(np.float32)
        self.task_active = np.ones(self.num_tasks, dtype=bool)
        self.task_assignments = {t: [] for t in range(self.num_tasks)}  # Track assigned robots
        debug_print(f"Object Positions:\n{self.object_positions}")
        debug_print(f"Target Positions:\n{self.target_positions}")
        debug_print(f"Task Active Flags:\n{self.task_active}")

        # Calculate required robots per task based on weights
        self.task_required_robots = self.task_weights.copy()
        debug_print(f"Task Required Robots: {self.task_required_robots}")

        # Calculate agent-to-task and agent-to-target distances
        self.agent_to_task_distances = np.linalg.norm(
            self.robot_positions[:, np.newaxis, :] - self.object_positions[np.newaxis, :, :],
            axis=2
        )  # Shape: (num_robots, num_tasks)
        self.agent_to_target_distances = np.linalg.norm(
            self.robot_positions[:, np.newaxis, :] - self.target_positions[np.newaxis, :, :],
            axis=2
        )  # Shape: (num_robots, num_tasks)
        debug_print(f"Agent to Task Distances:\n{self.agent_to_task_distances}")
        debug_print(f"Agent to Target Distances:\n{self.agent_to_target_distances}")

        obs = self.get_global_state()
        info = {}  # Additional info can be added here if needed

        # Update observation normalization parameters if enabled
        if self.normalize_observations:
            self.obs_mean = 0.9 * self.obs_mean + 0.1 * obs
            self.obs_std = 0.9 * self.obs_std + 0.1 * np.maximum(np.abs(obs - self.obs_mean), 1e-3)

        return obs, info

    def step(self, action):
        self.steps += 1
        rewards = {f"robot_{i}": self.time_penalty for i in range(self.num_robots)}  # Initialize with time penalty
        dones = {f"robot_{i}": False for i in range(self.num_robots)}
        infos = {f"robot_{i}": {} for i in range(self.num_robots)}

        # Split the flattened action array into per-agent actions
        actions_dict = {}
        for i in range(self.num_robots):
            start = i * 3
            end = (i + 1) * 3
            agent_id = f"robot_{i}"
            actions_dict[agent_id] = action[start:end]
        debug_print(f"Actions Dict: {actions_dict}")

        # Update robot velocities based on actions
        for i, agent_id in enumerate(sorted(actions_dict.keys())):
            agent_action = actions_dict[agent_id]
            desired_vx = np.clip(agent_action[0], -1.0, 1.0) * self.max_velocity
            desired_vy = np.clip(agent_action[1], -1.0, 1.0) * self.max_velocity
            desired_w  = np.clip(agent_action[2], -1.0, 1.0) * self.max_angular_velocity

            self.robot_velocities[i] = [desired_vx, desired_vy]
            self.robot_angular_vel[i] = desired_w
        debug_print(f"Updated Robot Velocities:\n{self.robot_velocities}")
        debug_print(f"Updated Robot Angular Velocities:\n{self.robot_angular_vel}")

        # Integrate positions and orientations
        dt = 0.1  # Time step
        previous_agent_to_task_distances = self.agent_to_task_distances.copy()
        previous_agent_to_target_distances = self.agent_to_target_distances.copy()
        for i in range(self.num_robots):
            self.robot_positions[i] += self.robot_velocities[i] * dt
            self.robot_positions[i] = np.clip(self.robot_positions[i], 0, self.field_size)
            self.robot_orientations[i] += self.robot_angular_vel[i] * dt
            self.robot_orientations[i] = ((self.robot_orientations[i] + np.pi) % (2 * np.pi)) - np.pi  # Keep within [-pi, pi]
        debug_print(f"Integrated Robot Positions:\n{self.robot_positions}")
        debug_print(f"Integrated Robot Orientations:\n{self.robot_orientations}")

        # Update agent-to-task and agent-to-target distances after movement
        self.agent_to_task_distances = np.linalg.norm(
            self.robot_positions[:, np.newaxis, :] - self.object_positions[np.newaxis, :, :],
            axis=2
        )
        self.agent_to_target_distances = np.linalg.norm(
            self.robot_positions[:, np.newaxis, :] - self.target_positions[np.newaxis, :, :],
            axis=2
        )
        debug_print(f"Updated Agent to Task Distances:\n{self.agent_to_task_distances}")
        debug_print(f"Updated Agent to Target Distances:\n{self.agent_to_target_distances}")

        # Reset task assignments
        self.task_assignments = {t: [] for t in range(self.num_tasks)}
        debug_print("Reset Task Assignments")

        # Assign robots to tasks based on proximity to objects
        for i in range(self.num_robots):
            for t in range(self.num_tasks):
                if self.task_active[t]:
                    if self.agent_to_task_distances[i, t] < self.completion_radius:
                        self.task_assignments[t].append(i)
                        # Reward for moving closer to the task
                        rewards[f"robot_{i}"] += (self.completion_radius - self.agent_to_task_distances[i, t]) * 0.1  # Shaped reward
                        debug_print(f"Robot {i} is close to Task {t} (Distance: {self.agent_to_task_distances[i, t]:.2f})")
        debug_print(f"Task Assignments: {self.task_assignments}")

        # Compute rewards and handle task completions
        for t in range(self.num_tasks):
            if self.task_active[t] and len(self.task_assignments[t]) >= self.task_required_robots[t]:
                # Reward all robots assigned to this task
                for robot_id in self.task_assignments[t]:
                    rewards[f"robot_{robot_id}"] += 5.0  # Intermediate reward for task completion
                self.task_active[t] = False
                self.tasks_completed += 1
                debug_print(f"Task {t} completed by robots {self.task_assignments[t]}")

        # Apply collision penalties if enabled
        if self.collision_penalty:
            # Simple collision detection based on overlapping positions
            for i in range(self.num_robots):
                for j in range(i + 1, self.num_robots):
                    dist = np.linalg.norm(self.robot_positions[i] - self.robot_positions[j])
                    if dist < 2 * self.robot_radius:
                        rewards[f"robot_{i}"] -= 1.0  # Increased collision penalty
                        rewards[f"robot_{j}"] -= 1.0
                        debug_print(f"Collision detected between robot_{i} and robot_{j}")

        # Penalize idling: agents with negligible velocity receive a small penalty
        movement_threshold = 0.1  # Threshold below which movement is considered idling
        for i in range(self.num_robots):
            movement = np.linalg.norm(self.robot_velocities[i])
            if movement < movement_threshold:
                rewards[f"robot_{i}"] -= 0.1  # Penalty for idling
                debug_print(f"Robot {i} is idling with movement {movement:.2f}")

        # Check termination conditions
        all_tasks_done = self.tasks_completed >= self.num_tasks
        terminated = all_tasks_done
        truncated = self.steps >= self.max_episode_steps
        done = terminated or truncated
        debug_print(f"Step {self.steps}: terminated={terminated}, truncated={truncated}, done={done}")

        # Generate observation
        obs = self.get_global_state()

        # Compute sum of rewards
        sum_rewards = sum(rewards.values())

        # ==========================
        # Reward Normalization
        # ==========================
        if self.normalize_rewards:
            scaling_factor = 100.0  # Adjust based on observed reward ranges
            normalized_sum_rewards = sum_rewards / scaling_factor
            normalized_sum_rewards = np.clip(normalized_sum_rewards, -100, 100)  # Ensure within bounds
        else:
            normalized_sum_rewards = sum_rewards

        # Update the total reward
        self.total_reward += normalized_sum_rewards
        debug_print(f"Sum of Rewards this step: {normalized_sum_rewards}, Total Reward: {self.total_reward}")

        # Info dict can contain per-agent rewards and tasks completed
        info = {
            "per_agent_rewards": rewards,
            "tasks_completed": self.tasks_completed
        }

        # Update observation normalization parameters if enabled
        if self.normalize_observations:
            self.obs_mean = 0.9 * self.obs_mean + 0.1 * obs
            self.obs_std = 0.9 * self.obs_std + 0.1 * np.maximum(np.abs(obs - self.obs_mean), 1e-3)
            # Normalize the observation
            obs = (obs - self.obs_mean) / self.obs_std

        # Return observation, normalized reward, terminated, truncated, info
        return obs, normalized_sum_rewards, terminated, truncated, info

    def get_global_state(self):
        """
        Concatenates and returns the global state, including all robots' positions, orientations,
        velocities, angular velocities, task states, and distances from each robot to each task.
        """
        # Flatten robot positions and velocities
        robots_flat = (
            self.robot_positions.flatten().tolist() +
            self.robot_orientations.tolist() +
            self.robot_velocities.flatten().tolist() +
            self.robot_angular_vel.tolist()
        )
        
        # Flatten task positions, target positions, active flags, and required robots
        tasks_flat = (
            self.object_positions.flatten().tolist() +
            self.target_positions.flatten().tolist() +
            self.task_active.astype(float).tolist() +
            self.task_required_robots.tolist()
        )
        
        # Flatten agent-to-task and agent-to-target distances
        distances_flat = (
            self.agent_to_task_distances.flatten().tolist() +
            self.agent_to_target_distances.flatten().tolist()
        )
        
        # Combine robots, tasks, and distances into a single global state list
        global_state = robots_flat + tasks_flat + distances_flat
        debug_print(f"Global State Shape: {len(global_state)}")
        
        return np.array(global_state, dtype=np.float32)

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
            self.ax.set_title("Collaborative Transport Environment")

            # Initialize robot patches and orientation arrows
            self.robot_patches = []
            self.orientation_arrows = []
            for i in range(self.num_robots):
                robot = patches.Circle((0,0), self.robot_radius, fc='blue', ec='black', alpha=0.6)
                self.robot_patches.append(robot)
                self.ax.add_patch(robot)
                # Arrow for orientation
                arrow = self.ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='red', ec='red')
                self.orientation_arrows.append(arrow)

            # Initialize object and target patches
            self.object_patches = []
            self.target_patches = []
            for t in range(self.num_tasks):
                obj = patches.Circle((0,0), 0.2, fc='orange', ec='black', alpha=0.8)
                self.object_patches.append(obj)
                self.ax.add_patch(obj)

                target = patches.Circle((0,0), 0.2, fc='green', ec='black', alpha=0.4)
                self.target_patches.append(target)
                self.ax.add_patch(target)

            # Initialize task status patches
            self.active_tasks_patches = []
            self.inactive_tasks_patches = []
            for t in range(self.num_tasks):
                status_patch = patches.Circle((0,0), 0.2, fc='green', ec='black', alpha=0.6)
                self.active_tasks_patches.append(status_patch)
                self.ax.add_patch(status_patch)
            self.fig.canvas.draw()

            # Initialize text elements
            self.text_elements['total_reward'] = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
            self.text_elements['steps'] = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes)
            self.text_elements['tasks_completed'] = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes)

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
            arrow = self.ax.arrow(pos[0], pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='red', ec='red')
            self.orientation_arrows[i] = arrow

        # Update object and target positions
        for t in range(self.num_tasks):
            obj_pos = self.object_positions[t]
            target_pos = self.target_positions[t]
            self.object_patches[t].center = (obj_pos[0], obj_pos[1])
            self.target_patches[t].center = (target_pos[0], target_pos[1])

        # Update task status patches based on activity
        # First, remove all status patches
        for patch in self.active_tasks_patches + self.inactive_tasks_patches:
            patch.remove()
        self.active_tasks_patches = []
        self.inactive_tasks_patches = []
        for t in range(self.num_tasks):
            tx, ty = self.object_positions[t]
            if self.task_active[t]:
                status_patch = patches.Circle((tx, ty), 0.2, fc='green', ec='black', alpha=0.6)
                self.active_tasks_patches.append(status_patch)
                self.ax.add_patch(status_patch)
            else:
                status_patch = patches.Circle((tx, ty), 0.2, fc='gray', ec='black', alpha=0.3)
                self.inactive_tasks_patches.append(status_patch)
                self.ax.add_patch(status_patch)

        # Update text elements
        self.text_elements['total_reward'].set_text(f'Total Reward: {self.total_reward:.2f}')
        self.text_elements['steps'].set_text(f'Steps: {self.steps}')
        self.text_elements['tasks_completed'].set_text(f'Tasks Completed: {self.tasks_completed} out of {self.num_tasks}')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig, self.ax = None, None
            debug_print("Closed the rendering window.")
