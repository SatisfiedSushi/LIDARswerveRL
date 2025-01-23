# utils/environment.py
from FullyAutoProject.Tests.PathPlanningBenchmarking.Simulations.dynamic_simulation import DynamicSimulation
from FullyAutoProject.Tests.PathPlanningBenchmarking.Simulations.static_simulation import StaticSimulation


class Environment:
    def __init__(self, sim_type, width=800, height=600,
                 static_obstacles=None, dynamic_obstacles=None, obstacle_speed=2,
                 start=(50, 50), goal=(750, 550), path=None, algorithm=None):
        """
        Initialize the simulation environment.

        :param sim_type: 'static' or 'dynamic'
        :param width: Width of the simulation window.
        :param height: Height of the simulation window.
        :param static_obstacles: List of static obstacle positions and sizes.
        :param dynamic_obstacles: List of dynamic obstacle dictionaries.
        :param obstacle_speed: Speed of dynamic obstacles.
        :param start: Start position tuple (x, y).
        :param goal: Goal position tuple (x, y).
        :param path: Optional path to visualize.
        :param algorithm: Path planning algorithm class for replanning (if needed).
        """
        if sim_type == 'static':
            if static_obstacles is None:
                static_obstacles = []
            self.simulation = StaticSimulation(width, height, static_obstacles, start, goal, path)
        elif sim_type == 'dynamic':
            if static_obstacles is None:
                static_obstacles = []
            if dynamic_obstacles is None:
                dynamic_obstacles = []
            self.simulation = DynamicSimulation(width, height, static_obstacles, dynamic_obstacles,
                                                obstacle_speed, start, goal, path, algorithm)
        else:
            raise ValueError("sim_type must be 'static' or 'dynamic'")

    def run(self):
        self.simulation.run()
