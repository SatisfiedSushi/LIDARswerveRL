# Simulations/static_simulation.py
from Tests.PathPlanningBenchmarking.utils.heuristic import heuristic


class StaticSimulation:
    def __init__(self, grid, start, goal, algorithm):
        """
        Initialize a static simulation.

        :param grid: 2D numpy array representing the environment.
        :param start: Tuple (x, y) for start position.
        :param goal: Tuple (x, y) for goal position.
        :param algorithm: Path planning algorithm class.
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.algorithm = algorithm

    def run(self):
        """
        Execute the simulation.

        :return: Path as a list of tuples, or None if no path found.
        """
        try:
            if self.algorithm.__name__ == "OkayPlan":
                # Define parameters for OkayPlan
                # Example parameters; replace with actual values as needed
                params = [1.0] * 19  # Adjust the number based on OkayPlan's requirements
                planner = self.algorithm(grid=self.grid, start=self.start, goal=self.goal, params=params)
                path, collision = planner.plan(env_info={
                    'start_point': self.start,
                    'target_point': self.goal,
                    'd2target': heuristic(self.start, self.goal),
                    'Obs_Segments': [],
                    'Flat_pdct_segments': []
                })
            else:
                planner = self.algorithm(grid=self.grid, start=self.start, goal=self.goal)
                path = planner.run()
                collision = False  # Static simulation assumes collision-free path
            return path, collision
        except Exception as e:
            import logging
            logging.error(f"{self.algorithm.__name__} encountered an error: {e}")
            return None, True  # Indicate a collision or failure
