import functools
import math
import random
import sys
import time
from copy import copy
# sys.setrecursionlimit(1000000)
import gymnasium as gym
from gymnasium.spaces.multi_discrete import MultiDiscrete
from gymnasium.spaces.box import Box
# from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import numpy as np
import pygame
import pygame._sdl2.controller
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.Box2D import *
from LIDAR import LIDAR
from SwerveDrive import SwerveDrive
from CoordConverter import CoordConverter

print(np.__version__)


class ScoreHolder:
    def __init__(self):
        self.red_points = 0
        self.blue_points = 0
        self.swerves = []

    def set_swerves(self, swerves):
        self.swerves = swerves

    def increase_points(self, team, robot):
        for swerve in self.swerves:
            if swerve.get_team() == team:
                swerve.set_score(swerve.get_score() + 1)
        match team:
            case 'Blue':
                self.blue_points += 1
            case 'Red':
                self.red_points += 1

    def reset_points(self):
        self.red_points = 0
        self.blue_points = 0

    # def render_score(self):
    #     font = pygame.font.Font(None, 36)
    #     score_text_red = font.render(f'Red Points: {self.red_points}', True, (255, 0, 0))
    #     score_text_blue = font.render(f'Blue Points: {self.blue_points}', True, (0, 0, 255))
    #     return score_text_red, score_text_blue

    def get_score(self, team):
        match team:
            case 'Blue':
                return self.blue_points
            case 'Red':
                return self.red_points


class MyContactListener(b2ContactListener):
    def destroy_body(self, body_to_destroy, team):
        body_to_destroy.userData = {"ball": True, 'Team': team, "isFlaggedForDelete": True}

    def GetBodies(self, contact):
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        body_a = fixture_a.body
        body_b = fixture_b.body

        return body_a, body_b

    def __init__(self, scoreHolder):
        b2ContactListener.__init__(self)
        self.scoreHolder = scoreHolder

    def BeginContact(self, contact):
        body_a, body_b = self.GetBodies(contact)
        main = None
        ball = None
        if body_a.userData is not None:
            main = body_a if 'robot' in body_a.userData else None
        if main is None:
            if body_b.userData is not None:
                main = body_b if 'robot' in body_b.userData else None
        if main is not None:
            if body_a.userData is not None:
                ball = body_a if 'ball' in body_a.userData else None
            if ball is None:
                if body_b.userData is not None:
                    ball = body_b if 'ball' in body_b.userData else None
            if ball is not None:
                new_ball_position = ((ball.position.x - main.position.x),
                                     (ball.position.y - main.position.y))

                angle_degrees = math.degrees(math.atan2(0 - new_ball_position[1],
                                                        0 - new_ball_position[0]) - np.pi)
                if angle_degrees < 0:
                    angle_degrees += 360

                if np.abs((math.degrees(main.angle) % 360) - angle_degrees) < 20:
                    '''print(main.angle)
                    print(angle_degrees)
                    print((math.degrees(main.angle) % 360) - angle_degrees)'''
                    # print("destroy")
                    if 'Team' in ball.userData:
                        self.scoreHolder.increase_points(ball.userData['Team'], main)

                    self.destroy_body(ball, ball.userData['Team'])

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


class env(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'name': 'Swerve-Env-V0'
    }
    render_mode = 'human'

    def pygame_to_box2d(self, pos_pygame):
        x, y = pos_pygame
        x_box2d = x
        y_box2d = -y  # Invert the y-coordinate to match PyBox2D's upward y-axis
        return x_box2d, y_box2d

    def box2d_to_pygame(self, pos_box2d):
        x, y = pos_box2d
        x_pygame = x
        y_pygame = -y  # Invert the y-coordinate to match Pygame's downward y-axis
        return x_pygame, y_pygame

    def meters_to_pixels(self, meters):
        return int(meters * self.PPM)

    # def sweep_dead_bodies(self):
    #     for body in self.world.bodies:
    #         if body is not None:
    #             data = body.userData
    #             if data is not None:
    #                 if "isFlaggedForDelete" in data:
    #                     if data["isFlaggedForDelete"]:
    #                         choice = random.randint(1, 4)
    #                         if 'ball' in body.userData and 'Team' in body.userData:
    #                             self.create_new_ball((self.hub_points[choice - 1].x, self.hub_points[choice - 1].y),
    #                                                  ((choice * (np.pi / 2)) + np.pi) + 1.151917 / 4, data["Team"])
    #                             self.balls.remove(body)
    #                             self.world.DestroyBody(body)
    #                             body.userData = None
    #                             body = None

    # def return_closest_ball(self, robot):
    #     LL_FOV = 31.65
    #     closest_ball = None
    #     angle_offset = 0
    #
    #     for ball in self.balls:
    #         if ball.userData['Team'] == robot.userData['Team']:
    #             new_ball_position = ((ball.position.x - robot.position.x),
    #                                  (ball.position.y - robot.position.y))
    #
    #             angle_degrees = math.degrees(math.atan2(0 - new_ball_position[1],
    #                                                     0 - new_ball_position[0]) - np.pi)
    #             if angle_degrees < 0:
    #                 angle_degrees += 360
    #             if np.abs((math.degrees(robot.angle) % 360) - angle_degrees) < LL_FOV:
    #                 if closest_ball is None:
    #                     closest_ball = ball
    #                     angle_offset = (math.degrees(robot.angle) % 360) - angle_degrees
    #                 elif (new_ball_position[0] ** 2 + new_ball_position[1] ** 2) < (
    #                         closest_ball.position.x ** 2 + closest_ball.position.y ** 2):
    #                     closest_ball = ball
    #                     angle_offset = (math.degrees(robot.angle) % 360) - angle_degrees
    #
    #     return closest_ball, angle_offset

    def return_robots_in_sight(self, robot_main):
        LL_FOV = 31.65  # 31.65 degrees off the center of the LL
        found_robots = []
        angles = []
        teams = []
        angle_offset = 0

        for robot in self.robots:
            new_robot_position = ((robot.position.x - robot_main.position.x),
                                  (robot.position.y - robot_main.position.y))

            angle_degrees = math.degrees(math.atan2(0 - new_robot_position[1],
                                                    0 - new_robot_position[0]) - np.pi)
            if angle_degrees < 0:
                angle_degrees += 360
            if np.abs((math.degrees(robot_main.angle) % 360) - angle_degrees) < LL_FOV:
                angle_offset = (math.degrees(robot_main.angle) % 360) - angle_degrees
                team = 0
                if robot.userData['Team'] == 'Blue':
                    team = 1
                else:
                    team = 2
                teams.append(team)
                angles.append(angle_offset)

        found_robots.append(teams)
        found_robots.append(angles)

        if len(found_robots[0]) != 0 and len(found_robots) != 5:
            for robot in range(5 - len(found_robots[0])):
                found_robots[0].append(0)
                found_robots[1].append(0)
        elif len(found_robots[0]) == 0:
            found_robots = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        return found_robots

    def create_new_ball(self, position, force_direction, team, force=0.014 - ((random.random() / 100))):
        x = position[0]
        y = position[1]

        new_ball = self.world.CreateDynamicBody(position=(x, y),
                                                userData={"ball": True,
                                                          "Team": team,
                                                          "isFlaggedForDelete": False})

        new_ball.CreateCircleFixture(radius=0.12, density=0.1, friction=0.001)
        friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_ball,
                                                bodyB=self.carpet,
                                                maxForce=0.01, maxTorque=5)
        self.world.CreateJoint(friction_joint_def)

        self.balls.append(new_ball)

        pos_or_neg = random.randint(0, 1)

        # force_direction = force_direction + (random.random()/36 if pos_or_neg == 0 else force_direction - random.random()) / 36 #  small random
        # force_direction = force_direction + (random.random()/18 if pos_or_neg == 0 else force_direction - random.random()) / 18 #  medium random
        force_direction = force_direction + (
            random.random() / 9 if pos_or_neg == 0 else force_direction - random.random()) / 9  # large random
        new_ball.ApplyLinearImpulse((np.cos(force_direction) * force, np.sin(force_direction) * force),
                                    point=new_ball.worldCenter, wake=True)

    def create_new_robot(self, **kwargs):
        position = kwargs['position'] or (0, 0)
        angle = kwargs['angle'] or 0
        team = kwargs['team'] or "Red"

        new_robot = self.world.CreateDynamicBody(position=position,
                                                 angle=angle,
                                                 userData={"robot": True,
                                                           "isFlaggedForDelete": False,
                                                           "Team": team})

        new_robot.CreatePolygonFixture(box=(0.56 / 2, 0.56 / 2), density=30, friction=0.01)
        friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_robot,
                                                bodyB=self.carpet,
                                                maxForce=10, maxTorque=10)
        self.world.CreateJoint(friction_joint_def)

        self.robots.append(new_robot)

    # def is_close_to_terminal(self, robot, red_spawned, blue_spawned):
    #     distance = 2.5
    #     force = 0.017
    #
    #     if robot.userData['Team'] == 'Blue':
    #         if math.sqrt(robot.position.x ** 2 + robot.position.y ** 2) < distance and not blue_spawned:
    #             self.create_new_ball(position=(0, 0), force_direction=np.pi / 4, team='Blue', force=force)
    #             return 'Blue'
    #
    #     else:
    #         if math.sqrt(robot.position.x ** 2 + robot.position.y ** 2) > -distance + math.sqrt(
    #                 self.terminal_red.position.x ** 2 + self.terminal_red.position.y ** 2) and not red_spawned:
    #             self.create_new_ball(position=(
    #                 self.terminal_red.position.x - (0.4 * math.sqrt(np.pi)),
    #                 self.terminal_red.position.y - (0.4 * math.sqrt(np.pi))),
    #                 force_direction=(np.pi / 4) + np.pi, team='Red', force=force)
    #             return 'Red'

    def __init__(self, render_mode="human"):
        super().__init__()
        # --- pygame setup ---
        # self.end_goal = ((random.randint(200, 1446) / 100), (random.randint(200, 623) / 100))
        self.end_goal = (5, 7)
        self.PPM = 100.0  # pixels per meter
        self.TARGET_FPS = 60
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self.meters_to_pixels(16.46), self.meters_to_pixels(8.23)
        self.screen = None
        self.clock = None
        self.teleop_time = 5  # 135 default
        self.CoordConverter = CoordConverter()

        # RL variables
        self.render_mode = render_mode
        # self.possible_agents = ["blue_1", "blue_2", "blue_3", "red_1", "red_2", "red_3"]
        # self.agent_ids = ["blue_1", "blue_2", "blue_3", "red_1", "red_2", "red_3"]
        self.possible_agents = ["blue_1"]
        self.agent_ids = ["blue_1"]
        self.agents = copy(self.possible_agents)
        self.resetted = False
        self.number_of_rays = 400
        # the end ray positions
        self.end_rays = []
        self.ray_distances = []
        self.ray_angles = []
        self.b2LIDARs = []
        self.distance_endpoints = []
        self.raycast_points = []
        self.observation_space = Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]), high=np.array([16.46, 8.23, 360, 16.46, 8.23]), shape=(5,))
        # self.observation_space = MultiDiscrete(np.array([1, 1]), seed=42)
        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,))

        self.W, self.A, self.S, self.D, self.LEFT, self.RIGHT = 0, 0, 0, 0, 0, 0

        self.red_Xs = None
        self.red_Ys = None
        self.red_angles = None
        self.red_LL_x_angles = None
        self.red_LL_robot_x_angles = None
        self.red_LL_robot_teams = None

        self.blue_Xs = None
        self.blue_Ys = None
        self.blue_angles = None
        self.blue_LL_x_angles = None
        self.blue_LL_robot_x_angles = None
        self.blue_LL_robot_teams = None

        self.previous_distance = 0

        self.timestep = None
        self.current_time = None
        self.game_time = None

        self.scoreHolder = None
        self.LIDAR = None

        # self.red_spawned = None
        # self.blue_spawned = None

        # --- other ---
        self.velocity_factor = 5
        self.angular_velocity_factor = 6
        # self.balls = None
        self.robots = None
        self.swerve_instances = None

        # --- Box2d ---
        self.world = None
        self.hub_points = None
        self.carpet = None
        self.carpet_fixture = None
        self.lower_wall = None
        self.left_wall = None
        self.right_wall = None
        self.upper_wall = None
        # self.terminal_blue = None
        # self.terminal_red = None
        # self.hub = None
        self.colors = {
            b2_staticBody: (255, 255, 255, 255),
            b2_dynamicBody: (127, 127, 127, 255),
        }

        def my_draw_polygon(polygon, body, fixture):
            vertices = [(body.transform * v) * self.PPM for v in polygon.vertices]
            vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]
            if body.userData is not None:
                pygame.draw.polygon(self.screen,
                                    (128, 128, 128, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255),
                                    vertices)
            else:
                pygame.draw.polygon(self.screen, self.colors[body.type], vertices)

        b2PolygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, fixture):
            position = body.transform * circle.pos * self.PPM
            position = (position[0], self.SCREEN_HEIGHT - position[1])
            pygame.draw.circle(self.screen, (0, 0, 255, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255),
                               [int(
                                   x) for x in position], int(circle.radius * self.PPM))
            # Note: Python 3.x will enforce that pygame get the integers it requests,
            #       and it will not convert from float.

        b2CircleShape.draw = my_draw_circle

    def reset(self, *, seed=None, options=None):

        # --- RL variables ---
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.scoreHolder = ScoreHolder()
        self.current_time = time.time()
        self.game_time = self.teleop_time - (time.time() - self.current_time)
        self.CoordConverter = CoordConverter()
        self.previous_distance = 0
        self.end_goal = (np.random.randint(140, 1500) / 100, np.random.randint(140, 750) / 100)
        # self.end_goal = (5, 7)

        # --- other ---
        # self.balls = []
        self.robots = []

        # self.red_spawned = False
        # self.blue_spawned = False

        # --- FRC game setup ---
        self.last_1sec_game_time = 0
        # self.hub_points = []
        self.world = b2World(gravity=(0, 0), doSleep=True, contactListener=MyContactListener(self.scoreHolder))
        self.LIDAR = LIDAR(self.world)
        self.number_of_rays = 100
        self.ray_angles = []
        self.b2LIDARs = []
        self.distance_endpoints = []
        self.raycast_points = []

        self.carpet = self.world.CreateStaticBody(
            position=(-3, -3),
        )

        '''self.carpet_fixture = self.carpet.CreatePolygonFixture(box=(1, 1), density=1, friction=0.1)  # frictopm = 0.3'''

        '''self.lower_wall = self.world.CreateStaticBody(
            position=(0, -1),
            shapes=b2PolygonShape(box=(16.46, 1)),
        )
        self.left_wall = self.world.CreateStaticBody(
            position=(-1 - 0.02, 0),  # -1 is the actual but it hurts my eyes beacuase i can still see the wall
            shapes=b2PolygonShape(box=(1, 8.23)),
        )
        self.right_wall = self.world.CreateStaticBody(
            position=(16.47 + 1, 0),  # 16.46 is the actual but it hurts my eyes beacuase i can still see the wall
            shapes=b2PolygonShape(box=(1, 8.23)),
        )
        self.upper_wall = self.world.CreateStaticBody(
            position=(0, 8.23 + 1),
            shapes=b2PolygonShape(box=(16.46, 1)),
        )'''

        '''self.sample_object = self.world.CreateStaticBody(
            position=(16.47/2, 8.23/2),
            shapes=b2PolygonShape(box=(1, 1)),
        )'''

        # self.terminal_blue = self.world.CreateStaticBody(
        #     position=((0.247) / math.sqrt(2), (0.247) / math.sqrt(2)),
        #     angle=np.pi / 4,
        #     shapes=b2PolygonShape(box=(0.99, 2.47)),
        # )
        #
        # self.terminal_red = self.world.CreateStaticBody(
        #     position=((16.46 - (0.247 / math.sqrt(2))), (8.23 - (0.247 / math.sqrt(2)))),
        #     angle=np.pi / 4,
        #     shapes=b2PolygonShape(box=(0.99, 2.47)),
        # )
        #
        # self.hub = self.world.CreateStaticBody(
        #     position=(16.46 / 2, 8.23 / 2),
        #     angle=1.151917,
        #     shapes=b2PolygonShape(box=(0.86, 0.86)),
        # )

        # for vertex in self.hub.fixtures[0].shape.vertices:
        #     new_vertex = self.hub.GetWorldPoint(vertex)
        #     offset = 0
        #     if new_vertex.x < 0:
        #         new_vertex.x -= offset
        #     else:
        #         new_vertex.x += offset
        #
        #     if new_vertex.y < 0:
        #         new_vertex.y -= offset
        #     else:
        #         new_vertex.y += offset
        #
        #     self.hub_points.append(new_vertex)

        ball_circle_diameter = 7.77
        ball_circle_center = (16.46 / 2, 8.23 / 2)

        # ball_x_coords = [0.658, -0.858, -2.243, -3.287, -3.790, -3.174, -0.658, 0.858, 2.243, 3.287, 3.790, 3.174,
        #                  -7.165,
        #                  7.165]
        # ball_y_coords = [3.830, 3.790, 3.174, 2.074, -0.858, -2.243, -3.830, -3.790, -3.174, -2.074, 0.858, 2.243,
        #                  -2.990,
        #                  2.990]
        #
        # ball_teams = ["Red", "Blue", "Red", "Blue", "Red", "Blue", "Blue", "Red", "Blue", "Red", "Blue", "Red", "Blue",
        #               "Red"]
        #
        # for x_coord, y_coord, team in zip(ball_x_coords, ball_y_coords, ball_teams):
        #     position = (x_coord + ball_circle_center[0], y_coord + ball_circle_center[1])
        #     self.create_new_ball(position=position, force_direction=0, team=team, force=0)

        robot_x_coords = [-1.3815]
        robot_y_coords = [0.5305]

        robot_teams = ["Blue"]

        for x_coord, y_coord, team in zip(robot_x_coords, robot_y_coords, robot_teams):
            position = (x_coord + ball_circle_center[0], y_coord + ball_circle_center[1])
            self.create_new_robot(position=position, angle=0, team=team)

        self.swerve_instances = [
            SwerveDrive(robot, robot.userData['Team'], 0, (1, 1), 1, velocity_factor=self.velocity_factor,
                        angular_velocity_factor=self.angular_velocity_factor) for robot in
            self.robots]  # TODO: find out how the fuck this works

        self.scoreHolder.set_swerves(swerves=self.swerve_instances)

        # create lidar instances

        '''observations = {
            agent: {
                'angle': np.array([0]),
                'angular_velocity': np.array([0]),
                'closest_ball': {
                    'angle': np.array([0]),
                },
                'robots_in_sight': {
                    'angles': np.array([0, 0, 0, 0, 0]),
                    'teams': np.array([0, 0, 0, 0, 0]),
                },
                'velocity': np.array([0, 0]),
            }

            for agent in self.agents
        }'''
        obs = np.array([0, 0, 0, 0, 0])

        self.reset_pygame()
        self.resetted = True

        infos = {}

        return obs, infos

    def reset_pygame(self):
        # --- pygame setup ---

        pygame.init()
        pygame.display.set_caption('Multi Agent Swerve Env')
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
        self.screen.fill((0, 0, 0, 0))

    def calculate_reward(self):
        # Constants for rewards and penalties
        TIME_STEP_PENALTY = -0.05
        PROGRESS_REWARD = 0.5
        STAGNATION_PENALTY = -0.2
        GOAL_ACHIEVEMENT_REWARD = 50.0
        OUT_OF_BOUNDS_PENALTY = -50.0
        DIRECTION_REWARD = 0.2  # Reward for facing towards the goal
        MAX_DISTANCE = 18.34  # Maximum possible distance to the goal

        terminated = False

        # Calculate distance to the goal
        goal_direction = (
            self.end_goal[0] - self.swerve_instances[0].get_box2d_instance().position.x,
            self.end_goal[1] - self.swerve_instances[0].get_box2d_instance().position.y
        )
        current_distance = math.hypot(*goal_direction)

        # Initialize reward with time step penalty
        reward = TIME_STEP_PENALTY

        # Reward for getting closer to the goal
        if current_distance < self.previous_distance:
            reward += PROGRESS_REWARD

        # Penalty for stagnation (not getting closer)
        if current_distance >= self.previous_distance:
            reward += STAGNATION_PENALTY

        # Reward for facing towards the goal
        # Assuming self.swerve_instances[0].angle represents the facing angle of the agent in radians
        agent_direction = (math.cos(self.swerve_instances[0].get_angle()), math.sin(self.swerve_instances[0].get_angle()))
        dot_product = sum(a * b for a, b in zip(goal_direction, agent_direction))
        angle_difference = math.acos(dot_product / (math.hypot(*goal_direction) * math.hypot(*agent_direction)))
        if angle_difference < math.pi / 4:  # within 45 degrees of the goal direction
            reward += DIRECTION_REWARD

        # Check if the goal is reached
        if current_distance < 0.05:  # Goal threshold
            reward += GOAL_ACHIEVEMENT_REWARD
            terminated = True

        # Penalty for going out of bounds
        if not (0 <= self.swerve_instances[0].get_box2d_instance().position.x <= 16.46 and
                0 <= self.swerve_instances[0].get_box2d_instance().position.y <= 8.23):
            reward += OUT_OF_BOUNDS_PENALTY
            terminated = True

        # Update previous distance for the next calculation
        self.previous_distance = current_distance

        return reward, terminated

    def step(self, actions):  # TODO: change action dictionary
        '''if not actions:
            # self.agents = []
            return {}, {}, {}, {}, {}'''

        self.game_time = self.teleop_time - (time.time() - self.current_time)
        # self.sweep_dead_bodies()
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.agents = []
                    pygame.quit()
                    return {}, {}, {}, {}, {}
                if event.key == pygame.K_w:
                    self.W = 1
                if event.key == pygame.K_a:
                    self.A = 1
                if event.key == pygame.K_s:
                    self.S = 1
                if event.key == pygame.K_d:
                    self.D = 1
                if event.key == pygame.K_LEFT:
                    self.LEFT = 1
                if event.key == pygame.K_RIGHT:
                    self.RIGHT = 1
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self.W = 0
                if event.key == pygame.K_a:
                    self.A = 0
                if event.key == pygame.K_s:
                    self.S = 0
                if event.key == pygame.K_d:
                    self.D = 0
                if event.key == pygame.K_LEFT:
                    self.LEFT = 0
                if event.key == pygame.K_RIGHT:
                    self.RIGHT = 0


        swerve = self.swerve_instances[0]

        '''lidar_holder = []
        for lidar in self.b2LIDARs:
            # destroys the lidar and creates a new body
            lid = self.b2LIDARs.index(lidar)
            new_lidar = self.world.CreateDynamicBody(
                position=self.swerve_instances[0].get_box2d_instance().position,
                angle=self.ray_angles[lid],
                shapes=b2PolygonShape(box=(4 if self.ray_distances[lid] == 0 else self.ray_distances[lid], 0.1)),
            )
            self.world.DestroyBody(lidar)

            for fixture in new_lidar.fixtures:
                fixture.sensor = True

            lidar_holder.append(new_lidar)

        self.b2LIDARs = lidar_holder'''

        # swerve = self.swerve_instances[self.agents.index(agent)]
        # swerve.set_velocity((actions[agent][0], actions[agent][1]))
        # swerve.set_angular_velocity(actions[agent][2])
        # swerve.set_velocity((0,0))
        # swerve.set_angular_velocity(0)
        # swerve.set_velocity((self.D - self.A, self.W - self.S))
        # swerve.set_angular_velocity(self.LEFT - self.RIGHT)
        swerve.set_velocity((actions[0], actions[1]))
        swerve.set_angular_velocity(actions[2])
        swerve.update()

        # match self.is_close_to_terminal(swerve.get_box2d_instance(), self.red_spawned, self.blue_spawned):
        #     case 'Red':
        #         self.red_spawned = True
        #     case 'Blue':
        #         self.blue_spawned = True

        self.world.Step(self.TIME_STEP, 10, 10)

        '''rewards = {
            agent:
                self.scoreHolder.get_score(
                    self.swerve_instances[self.agents.index(agent)].get_box2d_instance().userData['Team'])
            for agent in self.agents
        }'''

        # rewards from swerve isntances
        # rewards = {
        #     agent:
        #         (self.swerve_instances[self.agents.index(agent)].get_score() if not self.swerve_instances[self.agents.index(agent)].get_score_checked() else 0) + (np.abs(self.return_closest_ball(self.swerve_instances[self.agents.index(agent)].get_box2d_instance())[1]) / 18000)
        #     for agent in self.agents
        # }

        terminated = False
        truncated = False
        if self.game_time < 0:
            truncated = True

        # # more reward if current distance is less than previous distance else, negative reward
        # distance = math.hypot(self.end_goal[0] - self.swerve_instances[0].get_box2d_instance().position.x,
        #                       self.end_goal[1] - self.swerve_instances[0].get_box2d_instance().position.y)
        # rewards = -0.01  # small negative reward for each timestep
        #
        # # reward based on distance to the goal (closer is better)
        # rewards += 1.0 - (distance / 18.34)
        #
        # if distance < 0.1:
        #     rewards += self.game_time / 10  # reward proportional to remaining game time
        #
        #     # if near enough to the goal, give big reward and terminate
        #     if distance < 0.05:  # adjust this value as needed
        #         rewards += 100.0  # big reward
        #         terminated = True
        #
        # # if out of bounds terminate
        # if self.swerve_instances[0].get_box2d_instance().position.x < 0 or self.swerve_instances[
        #     0].get_box2d_instance().position.x > 16.46 or self.swerve_instances[
        #     0].get_box2d_instance().position.y < 0 or self.swerve_instances[0].get_box2d_instance().position.y > 8.23:
        #     terminated = True
        #     rewards -= 100.0  # large negative reward for going out of bounds
        #
        # self.previous_distance = distance

        rewards, terminated = self.calculate_reward()

        '''obs = {
            agent: {
                'velocity': np.asarray(self.swerve_instances[self.agents.index(agent)].get_velocity(), dtype=float32),
                'angular_velocity': np.array([self.swerve_instances[self.agents.index(agent)].get_angular_velocity()],
                                             dtype=float32),
                'angle': np.array([self.swerve_instances[self.agents.index(agent)].get_angle()], dtype=float32),
                'closest_ball': {
                    'angle': np.array([
                        self.return_closest_ball(self.swerve_instances[self.agents.index(agent)].get_box2d_instance())[
                            1]], dtype=float32)
                },
                'robots_in_sight': {
                    'angles': np.array(self.return_robots_in_sight(
                        self.swerve_instances[self.agents.index(agent)].get_box2d_instance())[1], dtype=float32),
                    'teams': np.array(self.return_robots_in_sight(
                        self.swerve_instances[self.agents.index(agent)].get_box2d_instance())[0], dtype=int64)
                }
            }

            for agent in self.agents
        }'''
        obs = (self.swerve_instances[0].get_box2d_instance().position.x,
               self.swerve_instances[0].get_box2d_instance().position.y,
               self.swerve_instances[0].get_angle(),
               self.end_goal[0],
               self.end_goal[1])

        info = {}

        # print(f'rewards: {rewards}')
        # print(f'obs: {obs}')

        if truncated:
            self.agents = []
            print("quit")
            #pygame.quit()

        return obs, rewards, terminated, truncated, info

    def close(self):
        print("quit")
        #pygame.quit()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.screen.fill((0, 0, 0, 0))

        # draw the sample object
        '''for fixture in self.sample_object.fixtures:
            fixture.shape.draw(self.sample_object, fixture)'''

        # for fixture in self.hub.fixtures:
        #     fixture.shape.draw(self.hub, fixture)
        #
        # for fixture in self.terminal_red.fixtures:
        #     fixture.shape.draw(self.terminal_red, fixture)
        #
        # for fixture in self.terminal_blue.fixtures:
        #     fixture.shape.draw(self.terminal_blue, fixture)
        #
        # for ball in self.balls:
        #     # adjusted ball position where the robot is centered at (0, 0)
        #     for fixture in ball.fixtures:
        #         fixture.shape.draw(ball, fixture)
        # render LIDAR rays

        for agent in self.agents:
            swerve = self.swerve_instances[self.agents.index(agent)]
            for fixture in swerve.get_box2d_instance().fixtures:
                fixture.shape.draw(swerve.get_box2d_instance(), fixture)

        pygame.draw.circle(self.screen, ('red'), self.CoordConverter.box2d_to_pygame(self.end_goal), 7)

        game_time_font = pygame.font.SysFont('Arial', 30)
        # Get the mouse position
        # self.screen.blit(self.scoreHolder.render_score()[1], (10, 10))
        # self.screen.blit(self.scoreHolder.render_score()[0], (self.screen.get_width() - 180, 10))s
        if np.floor(self.game_time) != self.last_1sec_game_time:
            self.last_1sec_game_time = np.floor(self.game_time)
            print(np.floor(self.game_time))
        # self.screen.blit(game_time_font.render(f'{int(self.teleop_time - self.game_time)}', True, (255, 255, 255)),
        #                  (self.screen.get_width() / 2 - 20, 10))

        pygame.display.flip()
        self.clock.tick(self.TARGET_FPS)
