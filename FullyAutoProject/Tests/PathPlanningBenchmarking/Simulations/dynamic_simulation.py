# Simulations/dynamic_simulation.py
from FullyAutoProject.Tests.PathPlanningBenchmarking.Simulations.static_simulation import StaticSimulation
from FullyAutoProject.Tests.PathPlanningBenchmarking.utils.heuristic import heuristic


class DynamicSimulation:
    def __init__(self, grid, start, goal, algorithm, dynamic_obstacles, obstacle_speed=2):
        """
        Initialize a dynamic simulation.

        :param grid: 2D numpy array representing the environment.
        :param start: Tuple (x, y) for start position.
        :param goal: Tuple (x, y) for goal position.
        :param algorithm: Path planning algorithm class.
        :param dynamic_obstacles: List of dictionaries with obstacle positions and directions.
        :param obstacle_speed: Speed at which dynamic obstacles move.
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.algorithm = algorithm
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_speed = obstacle_speed

    def run(self):
        """
        Execute the simulation.

        :return: Path as a list of tuples, or None if no path found.
        """
        # Implement dynamic obstacle movement and replanning as needed.
        # For simplicity, we'll assume static obstacles in this basic implementation.
        try:
            if self.algorithm.__name__ == "OkayPlan":
                # Define parameters for OkayPlan
                params = [1.0] * 19  # Adjust the number based on OkayPlan's requirements
                planner = self.algorithm(grid=self.grid, start=self.start, goal=self.goal, params=params)
                # Here, you would update 'Flat_pdct_segments' based on dynamic obstacles
                # For simplicity, we'll pass empty lists
                path, collision = planner.plan(env_info={
                    'start_point': self.start,
                    'target_point': self.goal,
                    'd2target': heuristic(self.start, self.goal),
                    'Obs_Segments': [],
                    'Flat_pdct_segments': []
                })
            else:
                sim = StaticSimulation(
                    grid=self.grid,
                    start=self.start,
                    goal=self.goal,
                    algorithm=self.algorithm
                )
                path, collision = sim.run()
            return path, collision
        except Exception as e:
            import logging
            logging.error(f"{self.algorithm.__name__} encountered an error: {e}")
            return None, True  # Indicate a collision or failure
