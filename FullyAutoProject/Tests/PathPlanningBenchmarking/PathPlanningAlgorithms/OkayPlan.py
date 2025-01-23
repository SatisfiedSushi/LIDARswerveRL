# PathPlanningAlgorithms/OkayPlan.py

from collections import deque
import numpy as np
import torch
import logging

from FullyAutoProject.PathPlanningAlgorithms.DStarLiteUtils.OccupancyGridMap import OccupancyGridMap
from FullyAutoProject.Tests.PathPlanningBenchmarking.utils.heuristic import heuristic


class OkayPlan:
    '''
    OkayPlan: A real-time path planner for dynamic environment
    Author: Jinghao Xin, Jinwoo Kim, Shengjia Chu, and Ning Li

    Paper Web: https://www.sciencedirect.com/science/article/pii/S002980182401179X
    Code Web: https://github.com/XinJingHao/OkayPlan

    Cite this algorithm:
    @article{XinOkayPlan,
    title = {OkayPlan: Obstacle Kinematics Augmented Dynamic real-time path Planning via particle swarm optimization},
    journal = {Ocean Engineering},
    volume = {303},
    pages = {117841},
    year = {2024},
    issn = {0029-8018},
    doi = {https://doi.org/10.1016/j.oceaneng.2024.117841},
    url = {https://www.sciencedirect.com/science/article/pii/S002980182401179X},
    author = {Jinghao Xin and Jinwoo Kim and Shengjia Chu and Ning Li}}

    Only for non-commercial purposes
    All rights reserved
    '''
    def __init__(self, grid, start, goal, params):
        """
        Initialize the OkayPlan algorithm.

        :param grid: 2D numpy array representing the environment.
        :param start: Tuple (x, y) for start position.
        :param goal: Tuple (x, y) for goal position.
        :param params: List of parameters required by OkayPlan.
        """
        self.dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

        '''Hyperparameter Initialization'''
        self.params = torch.tensor(params, device=self.dvc)  # Ensure params is a torch tensor
        self.G = 8  # Number of Groups
        # Inertia Initialization for 8 Groups:
        self.w_init = self.params[0:self.G].unsqueeze(-1).unsqueeze(-1)
        self.w_end = (self.params[0:self.G] * self.params[self.G:(2 * self.G)]).unsqueeze(-1).unsqueeze(-1)
        self.Max_iterations = int(self.params[16])  # Max iterations per frame
        self.w_delta = (self.w_init - self.w_end) / self.Max_iterations  # (G,1,1)
        # Velocity Initialization for 8 Groups:
        self.v_limit_ratio = self.params[(2 * self.G):(3 * self.G)].unsqueeze(-1).unsqueeze(-1)  # (G,1,1)
        self.v_init_ratio = 0.7 * self.v_limit_ratio  # (G,1,1)
        # H Matrix, (4,G=8,1,1):
        self.Hyper = torch.ones((4, self.G), device=self.dvc)
        self.Hyper[1] = self.params[(3 * self.G):(4 * self.G)]
        self.Hyper[2] = self.params[(4 * self.G):(5 * self.G)]
        self.Hyper[3] = self.params[(5 * self.G):(6 * self.G)]
        self.Hyper = self.Hyper.unsqueeze(-1).unsqueeze(-1)

        '''Particle Related'''
        self.N, self.D = 8, 10  # Number of Groups, particles per group, and particle dimension
        self.arange_idx = torch.arange(self.G, device=self.dvc)  # Index constant
        self.Search_range = [0, grid.shape[1]]  # Search space of the Particles (Assuming grid.shape = (rows, cols))
        self.Random = torch.zeros((4, self.G, self.N, 1), device=self.dvc)
        self.Kinmtc = torch.zeros((4, self.G, self.N, self.D), device=self.dvc)  # [V,Pbest,Gbest,Tbest]
        self.Locate = torch.zeros((4, self.G, self.N, self.D), device=self.dvc)  # [0,X,X,X]

        '''Path Related'''
        self.NP = int(self.D / 2)  # Number of path endpoints per particle
        self.S = self.NP - 1  # Number of path segments per particle
        self.P = self.G * self.N * self.S  # Total number of path segments
        self.rd_area = 0.5 * (self.Search_range[1] - self.Search_range[0]) / self.S  # Random range for particle initialization
        self.Start_V = heuristic(start, goal)  # Euclidean distance from start to goal

        '''Auto Truncation'''
        # For info, check https://arxiv.org/abs/2308.10169
        self.AT = True
        self.TrucWindow = 20
        self.std_Trsd = self.params[17].item()  # Truncation threshold

        '''Dynamic Prioritized Initialization'''
        self.DPI = self.params[18].item()

        '''Occupancy Grid Map'''
        self.map = OccupancyGridMap(x_dim=grid.shape[0], y_dim=grid.shape[1], exploration_setting='8N')
        self.map.grid = grid.copy()

        '''Path Initialization'''
        self.Priori_Path = None
        self.Tbest_Collision_Free = False
        self.Tbest_values_deque = deque(maxlen=self.TrucWindow)

        '''Initialize Planner'''
        self.initialize_planner(start, goal)

    def _uniform_random(self, low, high, shape):
        '''Generate uniformly random number in [low, high) in 'shape' on self.dvc'''
        return (high - low) * torch.rand(shape, device=self.dvc) + low

    def initialize_planner(self, start, goal):
        '''Initialize the planner with start and goal positions'''
        self.x_start, self.y_start = start
        self.x_target, self.y_target = goal
        self.d2target = heuristic(start, goal)  # Euclidean distance

        # Precompute obstacle segments if any
        obstacle_segments = self.map.get_obstacle_segments()
        self.Obs_Segments = torch.tensor(obstacle_segments, dtype=torch.float32, device=self.dvc) if len(obstacle_segments) > 0 else torch.empty((0, 2, 2), dtype=torch.float32, device=self.dvc)
        self.Flat_pdct_segments = torch.empty((0, 2, 2), dtype=torch.float32, device=self.dvc)  # Placeholder for predicted segments

        # Prior Path Initialization
        self.Priori_Path = torch.tensor([
            self.x_start + (self.x_target - self.x_start) * i / (self.NP - 1) for i in range(self.NP)
        ], device=self.dvc)

    def _fix_apex(self):
        '''Fix the start and goal positions in the Locate tensor'''
        self.Locate[1:4, :, :, 0] = self.x_start
        self.Locate[1:4, :, :, self.NP] = self.y_start
        self.Locate[1:4, :, :, self.NP - 1] = self.x_target
        self.Locate[1:4, :, :, 2 * self.NP - 1] = self.y_target

    def _ReLocate(self):
        '''Initialize particle positions before each iteration'''
        if self.DPI:
            # Dynamic Prioritized Initialization
            Mid_points = torch.tensor([[(self.x_start + self.x_target) / 2], [(self.y_start + self.y_target) / 2]], device=self.dvc).repeat((1, self.NP)).reshape(self.D)
            self.Locate[1:4] = Mid_points + self._uniform_random(low=-self.d2target / 2, high=self.d2target / 2, shape=(self.G, self.N, self.D))
        else:
            # SEPSO Position Initialization
            RN = int(0.25 * self.N)
            self.Locate[1:4] = self._uniform_random(low=self.Search_range[0], high=self.Search_range[1], shape=(self.G, self.N, self.D))
            self.Locate[1:4, :, 0:RN] = self.Priori_Path + self._uniform_random(-self.rd_area, self.rd_area, (self.G, RN, self.D))

        # Clamp particle positions within the search range
        self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1])
        self._fix_apex()

        # Initialize velocities
        self.Kinmtc[0] = (self.Hyper * self._uniform_random(0, 1, (self.G, self.N, self.D)) * (self.Kinmtc - self.Locate)).sum(dim=0)
        self.Kinmtc[0].clip_(self.v_min, self.v_max)

        # Initialize best values
        self.Pbest_value = torch.ones((self.G, self.N), device=self.dvc) * float('inf')
        self.Gbest_value = torch.ones(self.G, device=self.dvc) * float('inf')
        self.Tbest_value = float('inf')

        # Initialize deque for auto-truncation
        self.Tbest_values_deque = deque(maxlen=self.TrucWindow)

    def _Cross_product_for_VectorSet(self, V_PM, V_PP):
        '''Compute cross product for vector sets'''
        return V_PM[:, :, :, 0] * V_PP[:, 1, None, None] - V_PM[:, :, :, 1] * V_PP[:, 0, None, None]

    def _Is_Seg_Intersection_PtoM(self, P, M):
        '''Check intersection between particle segments and obstacle segments'''
        V_PP = P[:, 1] - P[:, 0]  # (p, 2)
        V_PM = M - P[:, None, None, 0]  # (p, m, 2, 2)
        Flag1 = self._Cross_product_for_VectorSet(V_PM, V_PP).prod(dim=-1) < 0  # (p, m)

        V_MM = M[:, 1] - M[:, 0]  # (m, 2)
        V_MP = P - M[:, None, None, 0]  # (m, p, 2, 2)
        Flag2 = self._Cross_product_for_VectorSet(V_MP, V_MM).prod(dim=-1) < 0  # (m, p)
        return Flag1 * Flag2.T  # (p, m)

    def _get_Obs_Penalty(self):
        """
        Computes the obstacle penalties for each group and particle.

        Returns:
            Obs_penalty (torch.Tensor): Shape (G, N)
            Pdct_Obs_penalty (torch.Tensor): Shape (G, N)
        """
        # Convert particle positions to segments
        particles = self.Locate[1].clone()  # (G, N, D)

        # Extract start and end points for each segment
        start_points = torch.stack(
            (particles[:, :, 0:(self.NP - 1)], particles[:, :, self.NP:(2 * self.NP - 1)]),
            dim=-1
        )  # (G, N, S, 2)

        end_points = torch.stack(
            (particles[:, :, 1:self.NP], particles[:, :, (self.NP + 1):2 * self.NP]),
            dim=-1
        )  # (G, N, S, 2)

        # Stack to form segments
        Segments = torch.stack(
            (start_points, end_points),
            dim=-2
        )  # (G, N, S, 2, 2)

        # Flatten segments to (P, 2, 2) where P = G * N * S
        flatted_Segments = Segments.reshape((self.P, 2, 2))  # (P, 2, 2)

        # Compute intersections with static obstacle segments
        Intersect_Matrix = self._Is_Seg_Intersection_PtoM(flatted_Segments, self.Obs_Segments)  # (P, M)

        # Sum over obstacle segments (dim=1) to get total intersections per segment
        Obs_penalty = Intersect_Matrix.sum(dim=1).reshape((self.G, self.N, self.S)).sum(dim=2) * self.params[10]  # (G, N)

        # Handle intersections with predicted dynamic obstacle segments
        if self.Flat_pdct_segments is None or self.Flat_pdct_segments.numel() == 0:
            Pdct_Obs_penalty = torch.zeros_like(Obs_penalty)  # (G, N)
        else:
            Pdct_Intersect_Matrix = self._Is_Seg_Intersection_PtoM(flatted_Segments, self.Flat_pdct_segments)  # (P, M_pdct)
            Pdct_Obs_penalty = Pdct_Intersect_Matrix.sum(dim=1).reshape((self.G, self.N, self.S)).sum(dim=2) * self.params[50]  # (G, N)

        return Obs_penalty, Pdct_Obs_penalty

    def _get_fitness(self):
        """
        Computes the fitness for each group and particle.

        Returns:
            fitness (torch.Tensor): Shape (G, N)
        """
        # Compute lengths of each segment
        Segments_length = torch.sqrt(
            (self.Locate[1, :, :, 0:(self.NP - 1)] - self.Locate[1, :, :, 1:self.NP]).pow(2) +
            (self.Locate[1, :, :, self.NP:(2 * self.NP - 1)] - self.Locate[1, :, :, (self.NP + 1):2 * self.NP]).pow(2)
        )  # (G, N, S)

        # Total path length
        Path_length = Segments_length.sum(dim=-1)  # (G, N)

        # Length of the lead segment (from start to first segment)
        LeadSeg_length = Segments_length[:, :, 0]  # (G, N)

        # Compute obstacle penalties
        self.Obs_penalty, Pdct_Obs_penalty = self._get_Obs_Penalty()  # Both (G, N)

        # Fitness computation
        fitness = (
                Path_length +
                self.params[48] * self.d2target * (self.Obs_penalty ** self.params[49]) +
                self.params[50] * self.d2target * (Pdct_Obs_penalty ** self.params[51]) +
                self.params[52] * self.d2target * (LeadSeg_length < 1.5 * self.Start_V).float()
        )

        return fitness  # (G, N)

    def _iterate(self):
        '''Particle iteration and path planning'''
        ''' Step 0: Dynamic Prioritized Initialization'''
        self._ReLocate()

        for i in range(self.Max_iterations):
            if self.AT and (i > 0.2 * self.TrucWindow) and self.Tbest_Collision_Free and (torch.std(torch.tensor(self.Tbest_values_deque)) < self.std_Trsd):
                break

            ''' Step 1: Compute Fitness (G,N) '''
            fitness = self._get_fitness()

            ''' Step 2: Update Pbest, Gbest, Tbest '''
            # Pbest
            P_replace = (fitness < self.Pbest_value)  # (G,N)
            self.Pbest_value[P_replace] = fitness[P_replace]  # Update Pbest_value
            self.Kinmtc[1, P_replace] = self.Locate[1, P_replace]  # Update Pbest_particles

            # Gbest
            values, indices = fitness.min(dim=-1)
            G_replace = (values < self.Gbest_value)  # (G,)
            self.Gbest_value[G_replace] = values[G_replace]  # Update Gbest_value
            self.Kinmtc[2, G_replace] = self.Locate[3, G_replace, indices[G_replace]].clone()

            # Tbest
            flat_idx = fitness.argmin()
            idx_g, idx_n = flat_idx // self.N, flat_idx % self.N
            min_fitness = fitness[idx_g, idx_n]
            if min_fitness < self.Tbest_value:
                self.Tbest_value = min_fitness  # Update Tbest_value
                self.Kinmtc[3, idx_g, idx_n] = self.Locate[3, idx_g, idx_n].clone()  # Update Tbest_particle
                # Check if Tbest path is collision-free
                if self.Obs_penalty[idx_g, idx_n] == 0:
                    self.Tbest_Collision_Free = True
                else:
                    self.Tbest_Collision_Free = False
            self.Tbest_values_deque.append(self.Tbest_value.item())

            ''' Step 3: Update particle velocities '''
            self.Hyper[0] = self.w_init - self.w_delta * i  # Inertia factor decay
            self.Random[1:4] = torch.rand((3, self.G, self.N, 1), device=self.dvc)  # Load random numbers
            self.Kinmtc[0] = (self.Hyper * self.Random * (self.Kinmtc - self.Locate)).sum(dim=0)  # (G,N,D)
            self.Kinmtc[0].clip_(self.v_min, self.v_max)  # Clamp velocity range

            ''' Step 4: Update particle positions '''
            self.Locate[1:4] += self.Kinmtc[0]  # (3,G,N,D) + (G,N,D) = (3,G,N,D)
            self.Locate[1:4].clip_(self.Search_range[0], self.Search_range[1])  # Clamp position range
            self._fix_apex()

    def plan(self, env_info):
        ''' Get environment information -> Iterate to solve path -> Update prior path for next time -> Return current path'''

        self.x_start, self.y_start = env_info['start_point']
        self.x_target, self.y_target = env_info['target_point']
        self.d2target = env_info['d2target']

        # Ensure obstacle segments are on the correct device
        self.Obs_Segments = torch.tensor(env_info['Obs_Segments'], dtype=torch.float32, device=self.dvc) if len(env_info['Obs_Segments']) > 0 else torch.empty((0, 2, 2), dtype=torch.float32, device=self.dvc)
        self.Flat_pdct_segments = torch.tensor(env_info['Flat_pdct_segments'], dtype=torch.float32, device=self.dvc) if len(env_info['Flat_pdct_segments']) > 0 else torch.empty((0, 2, 2), dtype=torch.float32, device=self.dvc)

        self._iterate()

        # Update prior path for next iteration
        self.Priori_Path = self.Kinmtc[3, 0, 0].clone()  # [x0,x1,...,y0,y1,...], shape=(D,), on self.dvc

        # Extract path
        path = []
        for i in range(self.NP):
            x = self.Priori_Path[i].item()
            y = self.Priori_Path[self.NP + i].item()
            path.append((x, y))

        return path, self.Tbest_Collision_Free
