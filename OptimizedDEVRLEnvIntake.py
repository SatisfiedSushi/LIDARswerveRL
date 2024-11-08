# env.py

import json
import struct
from multiprocessing import shared_memory
from multiprocessing import Process
from multiprocessing import Manager

import pandas
import os
import sys
import time
import math
import random
from copy import copy
import logging
import numpy as np
import pygame
import gymnasium as gym
from gymnasium.spaces import Box

from Box2D import (
    b2World,
    b2PolygonShape,
    b2CircleShape,
    b2FrictionJointDef,
    b2_staticBody,
    b2_dynamicBody,
    b2ContactListener,
)

import zmq
import msgpack

from LIDAR import LIDAR
from SwerveDrive import SwerveDrive
from CoordConverter import CoordConverter
from visualization import visualization_main



class DepthEstimator:
    def __init__(self, screen_width, screen_height, fov_angle=120, ray_count=100):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fov_angle = fov_angle
        self.ray_count = ray_count

    def rotate_point(self, x, y, cx, cy, angle_degrees):
        radians = math.radians(angle_degrees)
        cos = math.cos(radians)
        sin = math.sin(radians)
        nx = cos * (x - cx) - sin * (y - cy) + cx
        ny = sin * (x - cx) + cos * (y - cy) + cy
        return nx, ny

    def create_box(self, position, size, angle_degrees):
        cx, cy = position
        half_width, half_height = size[0] / 2, size[1] / 2
        corners = [
            (cx - half_width, cy - half_height),
            (cx + half_width, cy - half_height),
            (cx + half_width, cy + half_height),
            (cx - half_width, cy + half_height)
        ]
        rotated_corners = [self.rotate_point(x, y, cx, cy, angle_degrees) for x, y in corners]
        return rotated_corners

    def create_circle(self, center, radius):
        return center, radius

    def ray_intersects_segment(self, ray_origin, ray_direction, seg_start, seg_end):
        ray_dx, ray_dy = ray_direction
        seg_dx = seg_end[0] - seg_start[0]
        seg_dy = seg_end[1] - seg_start[1]
        denominator = (-seg_dx * ray_dy + ray_dx * seg_dy)
        if denominator == 0:
            return None
        t = (-ray_dy * (ray_origin[0] - seg_start[0]) + ray_dx * (ray_origin[1] - seg_start[1])) / denominator
        u = (seg_dx * (ray_origin[1] - seg_start[1]) - seg_dy * (ray_origin[0] - seg_start[0])) / denominator
        if 0 <= t <= 1 and u >= 0:
            intersection_x = seg_start[0] + t * seg_dx
            intersection_y = seg_start[1] + t * seg_dy
            return (intersection_x, intersection_y), u
        return None

    def ray_intersects_circle(self, ray_origin, ray_direction, circle_center, circle_radius):
        ox, oy = ray_origin
        dx, dy = ray_direction
        cx, cy = circle_center
        oc_x = cx - ox
        oc_y = cy - oy
        t_closest = (oc_x * dx + oc_y * dy) / (dx * dx + dy * dy)
        closest_x = ox + t_closest * dx
        closest_y = oy + t_closest * dy
        dist_to_center = math.hypot(closest_x - cx, closest_y - cy)
        if dist_to_center > circle_radius:
            return None
        t_offset = math.sqrt(circle_radius ** 2 - dist_to_center ** 2)
        t1 = t_closest - t_offset
        t2 = t_closest + t_offset
        if t1 >= 0:
            intersection_x = ox + t1 * dx
            intersection_y = oy + t1 * dy
            return (intersection_x, intersection_y), t1
        elif t2 >= 0:
            intersection_x = ox + t2 * dx
            intersection_y = oy + t2 * dy
            return (intersection_x, intersection_y), t2
        else:
            return None

    def calculate_angle(self, start_pos, end_pos):
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def cast_rays(self, camera_pos, look_at_pos, boxes, circles, max_ray_distance=500):
        cam_angle = self.calculate_angle(camera_pos, look_at_pos)
        half_fov = self.fov_angle / 2
        rays = []
        for i in range(self.ray_count):
            ray_angle = cam_angle - half_fov + (i / (self.ray_count - 1)) * self.fov_angle
            ray_direction = (math.cos(math.radians(ray_angle)), math.sin(math.radians(ray_angle)))
            rays.append((ray_angle, ray_direction))
        ray_intersections = []
        for ray_angle, ray_direction in rays:
            closest_intersection = None
            closest_distance = max_ray_distance
            hit_object_id = None
            for object_id, box in enumerate(boxes):
                for i in range(len(box)):
                    seg_start = box[i]
                    seg_end = box[(i + 1) % len(box)]
                    intersection = self.ray_intersects_segment(camera_pos, ray_direction, seg_start, seg_end)
                    if intersection:
                        point, distance = intersection
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_intersection = point
                            hit_object_id = object_id
            for circle_id, (center, radius) in enumerate(circles):
                intersection = self.ray_intersects_circle(camera_pos, ray_direction, center, radius)
                if intersection:
                    point, distance = intersection
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_intersection = point
                        hit_object_id = circle_id + len(boxes)
            # Append as dictionary instead of tuple
            ray_intersections.append({
                'intersection': closest_intersection,
                'distance': closest_distance,
                'object_id': hit_object_id,
                'ray_angle': ray_angle
            })
        return ray_intersections, cam_angle

    def visualize(self, screen, boxes, circles, camera_pos, look_at_pos, ray_intersections, coord_converter):
        # Convert boxes to Pygame coordinates for rendering
        boxes_pygame = [[coord_converter.box2d_to_pygame(v) for v in box] for box in boxes]
        # Draw boxes
        for box in boxes_pygame:
            pygame.draw.polygon(screen, (0, 0, 0), box, 2)
        # Convert camera position and look_at_pos to Pygame coordinates
        camera_pos_pygame = coord_converter.box2d_to_pygame(camera_pos)
        look_at_pos_pygame = coord_converter.box2d_to_pygame(look_at_pos)
        pygame.draw.circle(screen, (255, 0, 0), camera_pos_pygame, 5)
        # Draw FOV lines
        cam_angle = self.calculate_angle(camera_pos, look_at_pos)
        half_fov = self.fov_angle / 2
        fov_left_angle = cam_angle - half_fov
        fov_right_angle = cam_angle + half_fov
        fov_length = 5  # Adjusted for world units
        left_x = camera_pos[0] + fov_length * math.cos(math.radians(fov_left_angle))
        left_y = camera_pos[1] + fov_length * math.sin(math.radians(fov_left_angle))
        right_x = camera_pos[0] + fov_length * math.cos(math.radians(fov_right_angle))
        right_y = camera_pos[1] + fov_length * math.sin(math.radians(fov_right_angle))
        left_pygame = coord_converter.box2d_to_pygame((left_x, left_y))
        right_pygame = coord_converter.box2d_to_pygame((right_x, right_y))
        pygame.draw.line(screen, (128, 128, 128), camera_pos_pygame, left_pygame, 1)
        pygame.draw.line(screen, (128, 128, 128), camera_pos_pygame, right_pygame, 1)
        # Draw ray intersections
        for ray in ray_intersections:
            intersection = ray["intersection"]
            if intersection and isinstance(intersection, tuple) and len(intersection) == 2:
                intersection_pygame = coord_converter.box2d_to_pygame(intersection)
                pygame.draw.line(screen, (255, 0, 0), camera_pos_pygame, intersection_pygame, 1)
                pygame.draw.circle(screen, (255, 0, 0), (int(intersection_pygame[0]), int(intersection_pygame[1])), 3)
            else:
                # Handle cases where intersection is None or malformed
                continue

    def run_depth_estimation(self, camera_positions, look_at_positions, boxes, circles):
        all_ray_intersections = []
        for camera_pos, look_at_pos in zip(camera_positions, look_at_positions):
            ray_intersections, cam_angle = self.cast_rays(camera_pos, look_at_pos, boxes, circles)
            all_ray_intersections.append((camera_pos, look_at_pos, ray_intersections, cam_angle))
        return all_ray_intersections

    def get_1d_perspective(self, ray_intersections):
        half_fov = self.fov_angle / 2
        object_perspective = {}

        prev_object_id = None
        start_angle = None

        for i, ray in enumerate(ray_intersections):
            object_id = ray["object_id"]
            # Normalize the ray_angle from [-half_fov, half_fov] to [-1, 1]
            ray_angle_normalized = (-half_fov + i * (self.fov_angle / (self.ray_count - 1))) / half_fov

            if object_id != prev_object_id:
                if prev_object_id is not None:
                    # Store the previous object with its start and end angles
                    object_name = f"obj{prev_object_id + 1}"
                    object_perspective[object_name] = (start_angle, ray_angle_normalized)
                start_angle = ray_angle_normalized  # New object's start angle

            prev_object_id = object_id

        if prev_object_id is not None:
            # Store the last object
            object_name = f"obj{prev_object_id + 1}"
            object_perspective[object_name] = (start_angle, ray_angle_normalized)

        return object_perspective


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
        if team == 'Blue':
            self.blue_points += 1
        elif team == 'Red':
            self.red_points += 1

    def get_score(self, team):
        return self.blue_points if team == 'Blue' else self.red_points


class MyContactListener(b2ContactListener):
    def __init__(self, scoreHolder):
        b2ContactListener.__init__(self)
        self.scoreHolder = scoreHolder

    def destroy_body(self, body_to_destroy, team):
        body_to_destroy.userData = {"ball": True, 'Team': team, "isFlaggedForDelete": True}

    def BeginContact(self, contact):
        body_a, body_b = contact.fixtureA.body, contact.fixtureB.body
        main = None
        ball = None
        for body in [body_a, body_b]:
            if body.userData and 'robot' in body.userData:
                main = body
            elif body.userData and 'ball' in body.userData:
                ball = body
        if main and ball:
            new_ball_position = (ball.position.x - main.position.x, ball.position.y - main.position.y)
            angle_degrees = math.degrees(math.atan2(-new_ball_position[1], -new_ball_position[0]) - np.pi) % 360
            if abs((math.degrees(main.angle) % 360) - angle_degrees) < 20:
                if 'Team' in ball.userData:
                    self.scoreHolder.increase_points(ball.userData['Team'], main)
                self.destroy_body(ball, ball.userData['Team'])


class env(gym.Env):
    metadata = {'render.modes': ['human'], 'name': 'Swerve-Env-V0'}

    def __init__(self, render_mode="human", max_teleop_time=1000, lock=None):
        super().__init__()
        if lock is None:
            manager = Manager()
            self.lock = manager.Lock()
        else:
            self.lock = lock
        self.PPM = 100.0  # Ensure PPM is 100.0
        self.TARGET_FPS = 60
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.SCREEN_WIDTH = int(16.46 * self.PPM)  # 1646 pixels
        self.SCREEN_HEIGHT = int(8.23 * self.PPM)  # 823 pixels
        self.screen = None
        self.clock = None
        self.teleop_time = max_teleop_time
        self.CoordConverter = CoordConverter()
        self.starting_balls = 1
        self.balls = []
        self.render_mode = render_mode
        self.possible_agents = ["blue_1"]
        self.agent_ids = ["blue_1"]
        self.agents = copy(self.possible_agents)
        self.resetted = False
        self.depth_estimator = DepthEstimator(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, fov_angle=120, ray_count=200)
        self.previous_angle_to_ball = 0
        self.number_of_rays = 400
        self.end_rays = []
        self.ray_distances = []
        self.ray_angles = []
        self.b2LIDARs = []
        self.distance_endpoints = []
        self.raycast_points = []
        self.observation_space = Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, -np.pi, 0, -np.inf, 0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, 1, np.inf, np.inf])
        )
        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,))
        self.W, self.A, self.S, self.D, self.LEFT, self.RIGHT = 0, 0, 0, 0, 0, 0
        self.previous_distance_to_ball = 0
        self.timestep = None
        self.current_time = None
        self.distances = None
        self.game_time = None
        self.scoreHolder = None
        self.LIDAR = None
        self.velocity_factor = 5
        self.angular_velocity_factor = 6
        self.robots = []
        self.swerve_instances = []
        self.world = None
        self.obstacles = []
        self.carpet = None
        self.colors = {b2_staticBody: (255, 255, 255, 255), b2_dynamicBody: (127, 127, 127, 255)}
        self.depth_estimator_boxes = []
        self.depth_estimator_circles = []
        self.depth_estimations = []

        # Initialize Shared Memory
        self.shm_size = self.calculate_shared_memory_size()
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=self.shm_size, name="my_shared_memory")
            logging.info(f"Shared memory created with name: {self.shm.name}, size: {self.shm_size} bytes")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name="my_shared_memory")
            logging.info(f"Shared memory connected with name: {self.shm.name}, size: {self.shm_size} bytes")

        def my_draw_polygon(polygon, body, fixture):
            vertices = [body.GetWorldPoint(v) for v in polygon.vertices]
            vertices = [self.CoordConverter.box2d_to_pygame(v) for v in vertices]
            if body.userData is not None and 'Team' in body.userData:
                color = (128, 128, 128, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255)
            else:
                color = (255, 255, 255, 255)
            pygame.draw.polygon(self.screen, color, vertices)

        b2PolygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, fixture):
            position = body.transform * circle.pos * self.PPM
            position = (position[0], self.SCREEN_HEIGHT - position[1])
            color = (0, 0, 255, 255) if body.userData and body.userData.get('Team') == 'Blue' else (255, 0, 0, 255)
            pygame.draw.circle(self.screen, color, (int(position[0]), int(position[1])), int(circle.radius * self.PPM))

        b2CircleShape.draw = my_draw_circle

    def calculate_shared_memory_size(self):
        # Calculate the size based on the maximum expected data
        num_cameras = 2
        num_rays = 200
        total_size = (
                12 +  # robot_position (x, y, orientation)
                4 +  # timestamp (float)
                4 +  # number_of_cameras
                num_cameras * (
                        8 +  # camera_pos (x, y)
                        8 +  # look_at_pos (x, y)
                        4 +  # number_of_rays
                        num_rays * 24  # per ray data: 24 bytes per ray
                )
        )
        return total_size

    def reset(self, *, seed=None, options=None):
        self.distances = []
        self.previous_angle_to_ball = 0
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.scoreHolder = ScoreHolder()
        self.current_time = time.time()
        self.game_time = self.teleop_time - (time.time() - self.current_time)
        self.CoordConverter = CoordConverter()
        self.previous_distance_to_ball = 0
        self.balls = []
        self.robots = []
        self.last_1sec_game_time = 0
        self.world = b2World(gravity=(0, 0), doSleep=True, contactListener=MyContactListener(self.scoreHolder))
        self.LIDAR = LIDAR(self.world)
        self.number_of_rays = 100
        self.ray_angles = []
        self.b2LIDARs = []
        self.distance_endpoints = []
        self.raycast_points = []
        self.carpet = self.world.CreateStaticBody(position=(-3, -3))
        self.obstacles = [self.create_obstacle((4, 3), (0.5, 0.5)), self.create_obstacle((6, 2), (1, 1))]
        # Create walls
        self.walls = []
        world_width = 16.46
        world_height = 8.23
        wall_thickness = 0.1  # 10 cm walls

        # Left wall at x=0
        left_wall = self.world.CreateStaticBody(
            position=(0, world_height / 2),
            shapes=b2PolygonShape(box=(wall_thickness / 2, world_height / 2)),
            userData={"wall": True}
        )
        self.walls.append(left_wall)

        # Right wall at x=world_width
        right_wall = self.world.CreateStaticBody(
            position=(world_width, world_height / 2),
            shapes=b2PolygonShape(box=(wall_thickness / 2, world_height / 2)),
            userData={"wall": True}
        )
        self.walls.append(right_wall)

        # Bottom wall at y=0
        bottom_wall = self.world.CreateStaticBody(
            position=(world_width / 2, 0),
            shapes=b2PolygonShape(box=(world_width / 2, wall_thickness / 2)),
            userData={"wall": True}
        )
        self.walls.append(bottom_wall)

        # Top wall at y=world_height
        top_wall = self.world.CreateStaticBody(
            position=(world_width / 2, world_height),
            shapes=b2PolygonShape(box=(world_width / 2, wall_thickness / 2)),
            userData={"wall": True}
        )
        self.walls.append(top_wall)
        ball_circle_center = (16.46 / 2, 8.23 / 2)
        robot_x_coords = [-1.3815]
        robot_y_coords = [0.5305]
        robot_teams = ["Blue"]
        for x_coord, y_coord, team in zip(robot_x_coords, robot_y_coords, robot_teams):
            position = (x_coord + ball_circle_center[0], y_coord + ball_circle_center[1])
            self.create_new_robot(position=position, angle=0, team=team)
        self.swerve_instances = [
            SwerveDrive(robot, robot.userData['Team'], 0, (1, 1), 1, velocity_factor=self.velocity_factor,
                       angular_velocity_factor=self.angular_velocity_factor) for robot in self.robots]
        self.scoreHolder.set_swerves(swerves=self.swerve_instances)
        for i in range(self.starting_balls):
            self.create_random_ball()
        obs = np.zeros(self.observation_space.shape)
        self.reset_pygame()
        self.resetted = True
        infos = {}
        return obs, infos

    def reset_pygame(self):
        pygame.init()
        pygame.display.set_caption('Multi Agent Swerve Env')
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
        self.screen.fill((0, 0, 0, 0))

    def create_obstacle(self, position, size, angle=0):
        x, y = position
        obstacle = self.world.CreateStaticBody(
            position=(x, y),
            angle=math.radians(angle),
            userData={"obstacle": True, "isFlaggedForDelete": False}
        )
        obstacle.CreatePolygonFixture(box=(size[0] / 2, size[1] / 2), density=0, friction=0)
        return obstacle

    def create_new_robot(self, **kwargs):
        position = kwargs.get('position', (0, 0))
        angle = kwargs.get('angle', 0)
        team = kwargs.get('team', "Red")
        new_robot = self.world.CreateDynamicBody(position=position, angle=angle,
                                                 userData={"robot": True, "isFlaggedForDelete": False, "Team": team})
        new_robot.CreatePolygonFixture(box=(0.56 / 2, 0.56 / 2), density=30, friction=0.01)
        friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_robot,
                                                bodyB=self.carpet,
                                                maxForce=10, maxTorque=10)
        self.world.CreateJoint(friction_joint_def)
        self.robots.append(new_robot)

    def create_new_ball(self, position, force_direction, team, force=0.014 - ((random.random() / 100))):
        x, y = position
        new_ball = self.world.CreateDynamicBody(position=(x, y),
                                                userData={"ball": True, "Team": team, "isFlaggedForDelete": False})
        new_ball.CreateCircleFixture(radius=0.12, density=0.1, friction=0.001)
        friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_ball,
                                                bodyB=self.carpet, maxForce=0.01, maxTorque=5)
        self.world.CreateJoint(friction_joint_def)
        self.balls.append(new_ball)

    def create_random_ball(self):
        x_range = (1, 15.46)
        y_range_top = (1, 2)
        y_range_bottom = (6.23, 7.23)
        x_position = random.uniform(*x_range)
        y_position = random.uniform(*y_range_top) if random.choice([True, False]) else random.uniform(*y_range_bottom)
        self.create_new_ball(position=(x_position, y_position), force_direction=0, team=random.choice(["Red", "Blue"]))

    def find_closest_ball(self):
        closest_ball = None
        min_distance = float('inf')
        robot_position = self.swerve_instances[0].get_box2d_instance().position
        for ball in self.balls:
            ball_position = ball.position
            distance = math.hypot(ball_position.x - robot_position.x, ball_position.y - robot_position.y)
            if distance < min_distance:
                min_distance = distance
                closest_ball = ball
        return closest_ball, min_distance

    def is_robot_aligned_with_ball(self, ball):
        robot = self.swerve_instances[0].get_box2d_instance()
        robot_angle = math.degrees(robot.angle) % 360
        ball_position = ball.position
        robot_position = robot.position
        angle_to_ball = math.degrees(math.atan2(ball_position.y - robot_position.y,
                                                ball_position.x - robot_position.x))
        angle_to_ball = angle_to_ball % 360
        relative_angle = (angle_to_ball - robot_angle + 360) % 360
        return abs(relative_angle) < 20

    def has_picked_up_ball(self, ball):
        # Define a threshold distance to consider the ball picked up
        pickup_threshold = 0.3  # meters
        if ball is None:
            return False
        robot = self.swerve_instances[0].get_box2d_instance()
        distance = math.hypot(ball.position.x - robot.position.x, ball.position.y - robot.position.y)
        return distance < pickup_threshold

    def calculate_angle_to_ball(self, ball):
        if ball is None:
            return 0
        robot = self.swerve_instances[0].get_box2d_instance()
        robot_position = robot.position
        ball_position = ball.position
        delta_x = ball_position.x - robot_position.x
        delta_y = ball_position.y - robot_position.y
        angle_to_ball = math.atan2(delta_y, delta_x)
        robot_angle = robot.angle
        angle_relative_to_robot = angle_to_ball - robot_angle
        angle_relative_to_robot = (angle_relative_to_robot + math.pi) % (2 * math.pi) - math.pi
        return angle_relative_to_robot

    def get_camera_position(self, robot_position, robot_angle, camera_offset):
        x_offset, y_offset = camera_offset
        offset_x_rotated = x_offset * math.cos(robot_angle) - y_offset * math.sin(robot_angle)
        offset_y_rotated = x_offset * math.sin(robot_angle) + y_offset * math.cos(robot_angle)
        camera_x = robot_position[0] + offset_x_rotated
        camera_y = robot_position[1] + offset_y_rotated
        return camera_x, camera_y

    def step(self, actions, testing_mode=False):
        self.game_time = self.teleop_time - (time.time() - self.current_time)
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
        if testing_mode:
            swerve.set_velocity((self.D - self.A, self.W - self.S))
            swerve.set_angular_velocity(self.LEFT - self.RIGHT)
        else:
            swerve.set_velocity((actions[0], actions[1]))
            swerve.set_angular_velocity(actions[2])
        swerve.update()
        self.world.Step(self.TIME_STEP, 10, 10)
        terminated = False
        truncated = self.game_time < 0
        rewards, terminated = self.calculate_reward()

        # Depth Estimation
        robot = self.swerve_instances[0].get_box2d_instance()
        robot_angle = robot.angle  # In radians

        # Camera positions for front and back
        camera_offsets = [(0.28, 0), (-0.28, 0)]  # Front and back offsets
        camera_positions = []
        look_at_positions = []
        for idx, offset in enumerate(camera_offsets):
            camera_x, camera_y = self.get_camera_position(robot.position, robot_angle, offset)
            camera_pos = (camera_x, camera_y)  # Keep in world coordinates
            if idx == 0:
                # Front camera
                look_at_x = camera_x + math.cos(robot_angle) * 5  # 5 units ahead
                look_at_y = camera_y + math.sin(robot_angle) * 5
            else:
                # Back camera, look backwards
                look_at_x = camera_x + math.cos(robot_angle + math.pi) * 5
                look_at_y = camera_y + math.sin(robot_angle + math.pi) * 5
            look_at_pos = (look_at_x, look_at_y)  # In world coordinates
            camera_positions.append(camera_pos)
            look_at_positions.append(look_at_pos)

        # Prepare boxes and circles for depth estimation
        boxes = []
        for obstacle in self.obstacles:
            vertices_box2d = [obstacle.transform * v for v in obstacle.fixtures[0].shape.vertices]
            boxes.append(vertices_box2d)
        # Include walls
        for wall in self.walls:
            vertices_box2d = [wall.transform * v for v in wall.fixtures[0].shape.vertices]
            boxes.append(vertices_box2d)
        circles = []
        for ball in self.balls:
            position = (ball.position.x, ball.position.y)
            radius = ball.fixtures[0].shape.radius
            circle = self.depth_estimator.create_circle(position, radius)
            circles.append(circle)

        # Run depth estimation for both cameras
        self.depth_estimations = self.depth_estimator.run_depth_estimation(
            camera_positions, look_at_positions, boxes, circles
        )
        self.depth_estimator_boxes = boxes
        self.depth_estimator_circles = circles

        robot = self.swerve_instances[0].get_box2d_instance()
        robot_position = (robot.position.x, robot.position.y)
        robot_angle = robot.angle  # In radians

        depth_data = {
            "robot_position": {"x": robot_position[0], "y": robot_position[1]},
            "robot_orientation": math.degrees(robot_angle),  # Convert to degrees for readability
            "timestamp": time.time(),
            "depth_estimates": []
        }

        for camera_pos, look_at_pos, ray_intersections, cam_angle in self.depth_estimations:
            rays = []
            for ray in ray_intersections:
                intersection = ray["intersection"]
                distance = ray["distance"]
                object_id = ray["object_id"]
                ray_angle = ray["ray_angle"]
                rays.append({
                    'intersection': intersection,
                    'distance': distance,
                    'object_id': object_id,
                    'ray_angle': ray_angle
                })
            depth_data["depth_estimates"].append({
                'camera_pos': camera_pos,
                'look_at_pos': look_at_pos,
                'cam_angle': cam_angle,
                'rays': rays
            })

        # Serialize depth_data and write to shared memory with synchronization
        self.lock.acquire()
        try:
            self.serialize_and_write_shared_memory(depth_data)
        finally:
            self.lock.release()

        # Prepare observation
        closest_ball, distance_to_ball = self.find_closest_ball()
        angle_to_ball = self.calculate_angle_to_ball(closest_ball)
        velocity_x, velocity_y = self.swerve_instances[0].get_velocity()
        average_lidar_distance = 0  # Placeholder, implement if needed
        obs = np.array([
            robot.position.x, robot.position.y, self.swerve_instances[0].get_angle(),
            velocity_x, velocity_y, distance_to_ball, angle_to_ball,
            int(self.has_picked_up_ball(closest_ball)), self.game_time, average_lidar_distance
        ])
        info = {}
        if self.has_picked_up_ball(closest_ball) and self.is_robot_aligned_with_ball(closest_ball):
            terminated = True
        if truncated:
            self.agents = []
            pygame.quit()
        return obs, rewards, terminated, truncated, info

    def serialize_and_write_shared_memory(self, depth_data):
        try:
            logging.debug(f"Depth data written to shared memory at time {depth_data['timestamp']}")

            buffer = bytearray()

            # Robot Position (x, y)
            buffer += struct.pack('ff', depth_data["robot_position"]["x"], depth_data["robot_position"]["y"])

            # Robot Orientation (in radians)
            buffer += struct.pack('f', math.radians(depth_data["robot_orientation"]))

            # Timestamp
            buffer += struct.pack('f', depth_data["timestamp"])

            # Number of Cameras
            num_cameras = len(depth_data["depth_estimates"])
            buffer += struct.pack('i', num_cameras)

            for cam in depth_data["depth_estimates"]:
                # Camera Position
                buffer += struct.pack('ff', cam["camera_pos"][0], cam["camera_pos"][1])
                # Look At Position
                buffer += struct.pack('ff', cam["look_at_pos"][0], cam["look_at_pos"][1])
                # Number of Rays
                num_rays = len(cam["rays"])
                buffer += struct.pack('i', num_rays)
                for ray in cam["rays"]:
                    # Intersection x, y
                    inter_x, inter_y = ray["intersection"] if ray["intersection"] else (float('inf'), float('inf'))
                    buffer += struct.pack('ff', inter_x, inter_y)
                    # Distance
                    buffer += struct.pack('f', ray["distance"])
                    # Object ID
                    buffer += struct.pack('i', ray["object_id"] if ray["object_id"] is not None else -1)
                    # Ray Angle (in radians)
                    buffer += struct.pack('f', ray["ray_angle"])

            # Write to shared memory
            shm_view = memoryview(self.shm.buf)
            shm_view[:len(buffer)] = buffer
            logging.debug("Depth data written to shared memory.")
        except struct.error as e:
            logging.error(f"Struct packing error: {e}")
        except Exception as e:
            logging.error(f"Error writing to shared memory: {e}")

    def calculate_reward(self):
        TIME_STEP_PENALTY = -0.05
        PROGRESS_REWARD = 0.5
        ALIGNMENT_REWARD = 0.4
        PICKUP_REWARD = 100.0
        OUT_OF_BOUNDS_PENALTY = -50.0
        ANGLE_REWARD = 0.2
        terminated = False
        reward = TIME_STEP_PENALTY
        closest_ball, distance_to_ball = self.find_closest_ball()
        if distance_to_ball < self.previous_distance_to_ball:
            reward += PROGRESS_REWARD
        if self.is_robot_aligned_with_ball(closest_ball):
            reward += ALIGNMENT_REWARD
        angle_to_ball = self.calculate_angle_to_ball(closest_ball)
        if abs(angle_to_ball) < abs(self.previous_angle_to_ball):
            reward += ANGLE_REWARD
        if self.has_picked_up_ball(closest_ball):
            reward += PICKUP_REWARD
            terminated = True
        robot_pos = self.swerve_instances[0].get_box2d_instance().position
        if not (0 <= robot_pos.x <= 16.46 and 0 <= robot_pos.y <= 8.23):
            reward += OUT_OF_BOUNDS_PENALTY
            terminated = True
        self.previous_distance_to_ball = distance_to_ball
        self.previous_angle_to_ball = angle_to_ball
        return reward, terminated

    def close(self):
        pygame.quit()
        if hasattr(self, 'shm') and self.shm:
            self.shm.close()
            self.shm.unlink()
            logging.info("Shared memory closed and unlinked.")

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return
        self.screen.fill((0, 0, 0, 0))
        swerve = self.swerve_instances[0]
        for ball in self.balls:
            for fixture in ball.fixtures:
                fixture.shape.draw(ball, fixture)
        for obstacle in self.obstacles:
            for fixture in obstacle.fixtures:
                fixture.shape.draw(obstacle, fixture)
        for agent in self.agents:
            swerve = self.swerve_instances[self.agents.index(agent)]
            for fixture in swerve.get_box2d_instance().fixtures:
                fixture.shape.draw(swerve.get_box2d_instance(), fixture)
        # Draw walls
        for wall in self.walls:
            for fixture in wall.fixtures:
                fixture.shape.draw(wall, fixture)
        # Use DepthEstimator for visualization
        for camera_pos, look_at_pos, ray_intersections, cam_angle in self.depth_estimations:
            self.depth_estimator.visualize(
                self.screen, self.depth_estimator_boxes, self.depth_estimator_circles,
                camera_pos, look_at_pos, ray_intersections, self.CoordConverter
            )

        pygame.display.flip()
        self.clock.tick(self.TARGET_FPS)

    def serialize_and_write_shared_memory(self, depth_data):
        try:
            buffer = bytearray()

            # Robot Position
            buffer += struct.pack('ff', depth_data["robot_position"]["x"], depth_data["robot_position"]["y"])
            # Robot Orientation
            buffer += struct.pack('f', depth_data["robot_orientation"])
            # Timestamp
            buffer += struct.pack('f', depth_data["timestamp"])
            # Number of Cameras
            num_cameras = len(depth_data["depth_estimates"])
            buffer += struct.pack('i', num_cameras)

            for cam in depth_data["depth_estimates"]:
                # Camera Position
                buffer += struct.pack('ff', cam["camera_pos"][0], cam["camera_pos"][1])
                # Look At Position
                buffer += struct.pack('ff', cam["look_at_pos"][0], cam["look_at_pos"][1])
                # Number of Rays
                num_rays = len(cam["rays"])
                buffer += struct.pack('i', num_rays)
                for ray in cam["rays"]:
                    # Intersection x, y
                    inter_x, inter_y = ray["intersection"] if ray["intersection"] else (float('inf'), float('inf'))
                    buffer += struct.pack('ff', inter_x, inter_y)
                    # Distance
                    buffer += struct.pack('f', ray["distance"])
                    # Object ID
                    buffer += struct.pack('i', ray["object_id"] if ray["object_id"] is not None else -1)
                    # Ray Angle
                    buffer += struct.pack('f', ray["ray_angle"])

            # Write to shared memory
            shm_view = memoryview(self.shm.buf)
            shm_view[:len(buffer)] = buffer
            logging.debug("Depth data written to shared memory.")
        except struct.error as e:
            logging.error(f"Struct packing error: {e}")
        except Exception as e:
            logging.error(f"Error writing to shared memory: {e}")

    def run_depth_estimation(self, camera_positions, look_at_positions, boxes, circles):
        all_ray_intersections = []
        for camera_pos, look_at_pos in zip(camera_positions, look_at_positions):
            ray_intersections, cam_angle = self.depth_estimator.cast_rays(camera_pos, look_at_pos, boxes, circles)
            all_ray_intersections.append((camera_pos, look_at_pos, ray_intersections, cam_angle))
        return all_ray_intersections

    def close_shared_memory(self):
        if hasattr(self, 'shm') and self.shm:
            self.shm.close()
            self.shm.unlink()
            logging.info("Shared memory closed and unlinked.")

    def __del__(self):
        self.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    env_instance = env(render_mode="human", max_teleop_time=1000000)
    obs, info = env_instance.reset()

    # try:
    #     while True:
    #         obs, reward, terminated, truncated, info = env_instance.step((1.0, 0.0, 0.1), testing_mode=True)
    #         env_instance.render()
    #         if terminated or truncated:
    #             obs, info = env_instance.reset()
    # except KeyboardInterrupt:
    #     logging.info("Simulation interrupted by user.")
    # finally:
    #     env_instance.close()
    #     print("Simulation ended.")

    while True:
        obs, reward, terminated, truncated, info = env_instance.step((1.0, 0.0, 0.1), testing_mode=True)
        env_instance.render()
        if terminated or truncated:
            obs, info = env_instance.reset()