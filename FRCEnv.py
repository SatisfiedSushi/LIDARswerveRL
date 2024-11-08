import os
import sys
import time
import math
import random
from copy import copy

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

from SwerveDrive import SwerveDrive
from CoordConverter import CoordConverter


class DepthEstimator:
    def __init__(self, fov_angle=120, ray_count=200):
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
            hit_object_type = None  # New: Keep track of object type
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
                            hit_object_type = 'obstacle'  # Assume boxes are obstacles
            for circle_id, (center, radius) in enumerate(circles):
                intersection = self.ray_intersects_circle(camera_pos, ray_direction, center, radius)
                if intersection:
                    point, distance = intersection
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_intersection = point
                        hit_object_id = circle_id + len(boxes)
                        hit_object_type = 'ball'  # Circles are balls
            ray_intersections.append((closest_intersection, closest_distance, hit_object_id, ray_angle, hit_object_type))
        return ray_intersections, cam_angle

    def run_depth_estimation(self, camera_positions, look_at_positions, boxes, circles):
        all_ray_intersections = []
        for camera_pos, look_at_pos in zip(camera_positions, look_at_positions):
            ray_intersections, cam_angle = self.cast_rays(camera_pos, look_at_pos, boxes, circles)
            all_ray_intersections.append((camera_pos, look_at_pos, ray_intersections, cam_angle))
        return all_ray_intersections

    def visualize(self, screen, boxes, circles, camera_pos, look_at_pos, ray_intersections, coord_converter):
        # Draw boxes
        for box in boxes:
            box_pygame = [coord_converter.box2d_to_pygame(v) for v in box]
            pygame.draw.polygon(screen, (0, 0, 0), box_pygame, 2)
        # Draw circles
        for center, radius in circles:
            center_pygame = coord_converter.box2d_to_pygame(center)
            pygame.draw.circle(screen, (0, 0, 255), center_pygame, int(radius * coord_converter.PPM), 2)
        # Convert camera position to Pygame coordinates
        camera_pos_pygame = coord_converter.box2d_to_pygame(camera_pos)
        # Draw rays
        for intersection, _, _, _, _ in ray_intersections:
            if intersection:
                intersection_pygame = coord_converter.box2d_to_pygame(intersection)
                pygame.draw.line(screen, (255, 0, 0), camera_pos_pygame, intersection_pygame, 1)
                pygame.draw.circle(screen, (255, 0, 0), intersection_pygame, 2)
            else:
                # Draw ray to max distance
                ray_direction = (
                    math.cos(math.radians(self.calculate_angle(camera_pos, look_at_pos))),
                    math.sin(math.radians(self.calculate_angle(camera_pos, look_at_pos)))
                )
                max_point = (
                    camera_pos[0] + ray_direction[0] * 10,  # 10 units ahead
                    camera_pos[1] + ray_direction[1] * 10
                )
                max_point_pygame = coord_converter.box2d_to_pygame(max_point)
                pygame.draw.line(screen, (255, 0, 0), camera_pos_pygame, max_point_pygame, 1)


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
            # Here we could handle collision-based ball pickup, but we now use the intake mechanism
            pass


class env(gym.Env):
    metadata = {'render.modes': ['human'], 'name': 'Swerve-Env-V0'}

    def __init__(self, render_mode="human", max_teleop_time=5):
        super().__init__()
        self.PPM = 100.0
        self.TARGET_FPS = 60
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.SCREEN_WIDTH = int(16.46 * self.PPM)
        self.SCREEN_HEIGHT = int(8.23 * self.PPM)
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
        self.previous_angle_to_ball = 0
        self.number_of_rays = 400
        self.end_rays = []
        self.ray_distances = []
        self.ray_angles = []
        self.picked_up_balls = 0  # Keep track of picked up balls
        self.previous_distance_to_ball = float('inf')  # Set to a large number initially
        self.previous_angle_to_ball = float('inf')

        # Intake settings
        self.intake_width = 0.5  # Width of the intake rectangle in meters
        self.intake_height = 0.2  # Height of the intake rectangle in meters
        self.intake_offset = (0.3, 0.0)  # Offset from the robot's center (x, y)
        self.intake_fov = math.radians(30)  # Intake FOV in radians

        # Initialize DepthEstimator
        self.depth_estimator = DepthEstimator(fov_angle=120, ray_count=200)

        # Observation and action spaces
        self.observation_space = self.define_observation_space()

        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,))
        self.W, self.A, self.S, self.D, self.LEFT, self.RIGHT = 0, 0, 0, 0, 0, 0
        self.previous_distance_to_ball = 0
        self.timestep = None
        self.current_time = None
        self.game_time = None
        self.scoreHolder = None
        self.velocity_factor = 5
        self.angular_velocity_factor = 6
        self.robots = None
        self.swerve_instances = None
        self.world = None
        self.obstacles = None
        self.carpet = None
        self.colors = {b2_staticBody: (255, 255, 255, 255), b2_dynamicBody: (127, 127, 127, 255)}
        self.depth_estimator_boxes = []
        self.depth_estimator_circles = []
        self.depth_estimations = []

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
            pygame.draw.circle(self.screen, (0, 0, 255, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255),
                               [int(x) for x in position], int(circle.radius * self.PPM))

        b2CircleShape.draw = my_draw_circle

    def define_observation_space(self):
        # Define observation sizes
        num_front_rays = 200
        num_back_rays = 200
        num_total_rays = num_front_rays + num_back_rays

        # Observation components:
        # 1) angle_to_ball: [-pi, pi]
        # 2) distance_to_ball: [0, 20]
        # 3) ball_picked_up_indicator: [0, 1]
        # 4) robot.position.x: [0, 17]
        # 5) robot.position.y: [0, 9]
        # 6) robot_angle: [-pi, pi]
        # 7) velocity_x: [-5, 5]
        # 8) velocity_y: [-5, 5]
        # 9) game_time: [0, 150]
        # 10) Front camera ray distances (200): [0, 10]
        # 11) Back camera ray distances (200): [0, 10]
        # 12) Additional observations (4):
        #     a) Distance to ball: [0, 20]
        #     b) Angle to ball: [-pi, pi]
        #     c) Ball picked up indicator: [0, 1]
        #     d) Game time remaining: [0, 150]

        obs_low = [
            -60,  # angle_to_ball
            0.0,  # distance_to_ball
            0.0,  # ball_picked_up_indicator
            0.0,  # robot.position.x
            0.0,  # robot.position.y
            0,  # robot_angle
            -1,  # velocity_x
            -1,  # velocity_y
            0.0  # game_time
        ]

        obs_high = [
            60,  # angle_to_ball
            np.inf,  # distance_to_ball
            1.0,  # ball_picked_up_indicator
            16.46,  # robot.position.x
            8.23,  # robot.position.y
            360,  # robot_angle
            1,  # velocity_x
            1,  # velocity_y
            150.0  # game_time
        ]

        # Front camera rays
        obs_low.extend([0.0] * num_front_rays)
        obs_high.extend([np.inf] * num_front_rays)

        # Back camera rays
        obs_low.extend([0.0] * num_back_rays)
        obs_high.extend([30] * num_back_rays)

        # Additional observations
        obs_low.extend([
            0.0,  # Distance to ball
            -60,  # Angle to ball
            0.0,  # Ball picked up indicator
            0.0  # Game time remaining
        ])

        obs_high.extend([
            30,  # Distance to ball
            60,  # Angle to ball
            1.0,  # Ball picked up indicator
            150.0  # Game time remaining
        ])

        self.observation_space = Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32
        )

        return self.observation_space

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
        self.number_of_rays = 100
        self.ray_angles = []
        self.distance_endpoints = []
        self.raycast_points = []
        self.carpet = self.world.CreateStaticBody(position=(-3, -3))
        self.obstacles = [self.create_obstacle((4, 3), (0.5, 0.5)), self.create_obstacle((6, 2), (1, 1))]
        self.previous_distance_to_ball = float('inf')  # Set to a large number initially
        self.previous_angle_to_ball = float('inf')
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
        self.picked_up_balls = 0  # Reset picked up balls count
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
        position = kwargs['position'] or (0, 0)
        angle = kwargs['angle'] or 0
        team = kwargs['team'] or "Red"
        new_robot = self.world.CreateDynamicBody(position=position, angle=angle,
                                                 userData={"robot": True, "isFlaggedForDelete": False, "Team": team})
        new_robot.CreatePolygonFixture(box=(0.56 / 2, 0.56 / 2), density=30, friction=0.01)
        friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_robot,
                                                bodyB=self.carpet,
                                                maxForce=10, maxTorque=10)
        self.world.CreateJoint(friction_joint_def)
        self.robots.append(new_robot)

    def create_new_ball(self, position, force_direction, team):
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
        # Get the raycast data from the front camera (assumed to be the first in self.depth_estimations)
        if not self.depth_estimations:
            return None, float('inf')

        front_camera_data = self.depth_estimations[0]
        camera_pos, look_at_pos, ray_intersections, cam_angle = front_camera_data

        detected_balls = []
        robot_position = self.swerve_instances[0].get_box2d_instance().position

        num_boxes = len(self.depth_estimator_boxes)

        for intersection, distance, object_id, ray_angle, hit_object_type in ray_intersections:
            if hit_object_type == 'ball' and intersection:
                # Adjust object_id to get the index in self.balls
                ball_index = object_id - num_boxes
                if 0 <= ball_index < len(self.balls):
                    ball = self.balls[ball_index]
                    # Compute the distance from the robot to the ball
                    ball_position = ball.position
                    dist = math.hypot(ball_position.x - robot_position.x, ball_position.y - robot_position.y)
                    detected_balls.append((ball, dist))

        if detected_balls:
            # Find the closest ball among detected balls
            closest_ball, min_distance = min(detected_balls, key=lambda x: x[1])
            return closest_ball, min_distance
        else:
            # No ball detected
            return None, float('inf')

    def is_robot_aligned_with_ball(self, ball):
        robot = self.swerve_instances[0].get_box2d_instance()
        robot_angle = math.degrees(robot.angle) % 360
        ball_position = ball.position
        robot_position = robot.position
        angle_to_ball = math.degrees(math.atan2(ball_position.y - robot_position.y,
                                                ball_position.x - robot_position.x))
        angle_to_ball = angle_to_ball % 360
        relative_angle = (angle_to_ball - robot_angle + 360) % 360
        threshold_angle = 70
        return abs(relative_angle) < threshold_angle

    def has_picked_up_ball(self):
        robot = self.swerve_instances[0].get_box2d_instance()
        robot_angle = robot.angle  # In radians
        robot_position = robot.position
        cos_angle = math.cos(-robot_angle)
        sin_angle = math.sin(-robot_angle)
        picked_up = False
        balls_to_remove = []
        for ball in self.balls:
            ball_position = ball.position
            # Translate ball position relative to robot
            dx = ball_position.x - robot_position.x
            dy = ball_position.y - robot_position.y
            # Rotate to robot's local frame
            local_x = dx * cos_angle - dy * sin_angle
            local_y = dx * sin_angle + dy * cos_angle
            # Check if within intake rectangle
            half_width = self.intake_width / 2
            half_height = self.intake_height / 2
            if (self.intake_offset[0] - half_width <= local_x <= self.intake_offset[0] + half_width and
                self.intake_offset[1] - half_height <= local_y <= self.intake_offset[1] + half_height):
                # Ball is within intake rectangle
                balls_to_remove.append(ball)
                picked_up = True
                self.picked_up_balls += 1
                # Spawn a new ball
                self.create_random_ball()
        # Remove picked up balls
        for ball in balls_to_remove:
            self.balls.remove(ball)
            self.world.DestroyBody(ball)
        return picked_up

    def calculate_angle_to_ball(self, ball):
        if ball is None:
            return None  # Return None if no ball exists

        # Get the robot instance and position
        robot = self.swerve_instances[0].get_box2d_instance()
        robot_position = robot.position
        ball_position = ball.position

        # Compute the angle to the ball from the robot's current position
        delta_x = ball_position.x - robot_position.x
        delta_y = ball_position.y - robot_position.y
        angle_to_ball = math.degrees(math.atan2(delta_y, delta_x))  # Angle to ball in world coordinates

        # Convert to robot-relative coordinates (robot's angle is 0 when it is facing forward)
        robot_angle = math.degrees(robot.angle)  # Convert robot's angle to degrees

        # Calculate the relative angle to the ball
        angle_relative_to_robot = angle_to_ball - robot_angle

        # Normalize angle to be within [-180, 180]
        angle_relative_to_robot = (angle_relative_to_robot + 180) % 360 - 180

        # Check if the ball is within the FOV of the intake camera (front camera)
        half_fov = math.degrees(120) / 2  # FOV is centered around 0 (the robot's forward direction)

        if -half_fov <= angle_relative_to_robot <= half_fov:
            # Ball is within the camera's FOV, return the relative angle
            return -angle_relative_to_robot
        else:
            # Ball is outside the FOV, return None or a default value like 0
            return 1000

    def get_distance_to_ball(self, ball):
        robot_position = self.swerve_instances[0].get_box2d_instance().position
        ball_position = ball.position
        distance = math.hypot(ball_position.x - robot_position.x, ball_position.y - robot_position.y)
        return distance if distance is not float('inf') else 1000

    def step(self, actions=None, testing_mode=False):
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
            # Use keyboard inputs for movement
            swerve.set_velocity((self.D - self.A, self.W - self.S))
            swerve.set_angular_velocity(self.LEFT - self.RIGHT)
        else:
            # Use actions from the agent
            swerve.set_velocity((actions[0], actions[1]))
            swerve.set_angular_velocity(actions[2])
        swerve.update()
        self.world.Step(self.TIME_STEP, 10, 10)
        terminated = False
        truncated = self.game_time < 0

        # Run depth estimation for both cameras
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
            for fixture in wall.fixtures:
                if isinstance(fixture.shape, b2PolygonShape):
                    vertices_box2d = [wall.transform * v for v in fixture.shape.vertices]
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

        # Find closest ball and calculate distance and angle to it
        closest_ball, distance_to_ball = self.find_closest_ball()
        angle_to_ball = self.calculate_angle_to_ball(closest_ball)

        if distance_to_ball != float('inf'):
            # Update previous distance and angle after reward calculation
            self.previous_distance_to_ball = distance_to_ball
            self.previous_angle_to_ball = abs(
                angle_to_ball) if angle_to_ball is not None else self.previous_angle_to_ball

        # Calculate reward including raycast penalties
        reward, terminated = self.calculate_reward()
        obs = self.construct_observation()
        info = {}
        if self.picked_up_balls >= 5:  # For example, terminate after picking up 5 balls
            terminated = True
        if truncated:
            self.agents = []
            pygame.quit()
        return obs, reward, terminated, truncated, info

    def construct_observation(self):
        closest_ball, distance_to_ball = self.find_closest_ball()
        angle_to_ball = self.calculate_angle_to_ball(closest_ball)

        # First 3 observations
        if angle_to_ball is not None:
            # Ball is in view
            observations = [angle_to_ball, distance_to_ball, 1.0]  # Ball picked up indicator = 1.0
        else:
            # Ball is not in view
            observations = [0.0, 0.0, 0.0]  # Ball not in view or picked up

        # Next 6 observations: robot state information
        robot = self.swerve_instances[0].get_box2d_instance()
        velocity_x, velocity_y = self.swerve_instances[0].get_velocity()
        # print(f'entire robot state by category: posx {robot.position.x}, posy {robot.position.y}, angle {self.swerve_instances[0].get_angle()}, velx {velocity_x}, vely {velocity_y}, game time {self.game_time}')
        robot_state = [
            robot.position.x,
            robot.position.y,
            self.swerve_instances[0].get_angle(),
            velocity_x,
            velocity_y,
            self.game_time  # Adding game time remaining to the observation
        ]
        observations.extend(robot_state)

        # Ray distances from front and back cameras
        for idx, (camera_pos, look_at_pos, ray_intersections, cam_angle) in enumerate(self.depth_estimations):
            distances = []
            for intersection, distance, object_id, ray_angle, hit_object_type in ray_intersections:
                if distance is not None and not math.isnan(distance):
                    distances.append(distance)  # Cap the distance at 10.0 meters
                else:
                    distances.append(30)  # Max distance if no intersection
            observations.extend(distances)

        # Additional observations (as per request)
        observations.extend([
            1000,     # Distance to ball (assuming 0)
            1000,  # Angle to ball (-π)
            0.0,     # Ball picked up indicator (0)
            0.0      # Game time remaining (0)
        ])
        # print(f"Observations: {observations}")  # Optionally print observations for debugging

        return np.array(observations, dtype=np.float32)

    def calculate_reward(self):
        # Constants
        PICKUP_REWARD = 50.0
        OUT_OF_BOUNDS_PENALTY = -10.0
        TIME_STEP_PENALTY = -0.1
        DISTANCE_FACTOR = 3.0  # Scaling factor for distance reward
        ANGLE_FACTOR = 0.5  # Scaling factor for angle reward
        LINE_OF_SIGHT_REWARD = 0.5
        MAX_LIDAR_DISTANCE = 10.0  # Maximum distance for lidar readings (meters)
        LIDAR_PENALTY_FACTOR = -0.5  # Scaling factor for lidar penalty
        LIDAR_REWARD_FACTOR = 0.1  # Scaling factor for lidar reward
        MAX_DISTANCE = 10.0  # Maximum distance to normalize the reward

        reward = 0.0
        terminated = False

        # Time step penalty
        reward += TIME_STEP_PENALTY

        # Get the closest ball and compute angle and distance
        closest_ball, distance_to_ball = self.find_closest_ball()

        # Ensure distance_to_ball is valid
        if distance_to_ball == float('inf') or math.isnan(distance_to_ball):
            distance_to_ball = MAX_DISTANCE  # Use max distance if no ball is found

        # Distance reward: inversely proportional to distance (closer gives higher reward)
        distance_reward = DISTANCE_FACTOR * (MAX_DISTANCE - distance_to_ball) / MAX_DISTANCE
        reward += distance_reward
        if distance_to_ball == 1000:
            distance_reward = 0

        # Get angle to ball
        angle_to_ball = self.calculate_angle_to_ball(closest_ball)

        if angle_to_ball is None or math.isnan(angle_to_ball) or angle_to_ball == float('inf'):
            angle_to_ball = np.pi  # Assume the worst-case angle (180 degrees) if invalid

        # Angle reward: inversely proportional to the angle (more aligned gives higher reward)

        normalized_angle = abs(angle_to_ball) / np.pi  # Normalize the angle between 0 and 1
        angle_reward = ANGLE_FACTOR * (1 - normalized_angle)  # The more aligned, the higher the reward
        if angle_to_ball == 1000:
            angle_reward = 0
        reward += angle_reward

        # Line of sight reward: small bonus if the ball is within the intake FOV
        half_fov = self.intake_fov / 2
        if abs(angle_to_ball) <= half_fov:
            reward += LINE_OF_SIGHT_REWARD

        # Lidar (Raycast) Penalty and Reward
        lidar_penalty = 0.0
        lidar_reward = 0.0
        num_rays = 0

        for camera_pos, look_at_pos, ray_intersections, cam_angle in self.depth_estimations:
            for intersection, distance, object_id, ray_angle, hit_object_type in ray_intersections:
                if distance is not None and not math.isnan(distance):
                    num_rays += 1
                    # Penalize for being close to obstacles
                    if distance < MAX_LIDAR_DISTANCE:
                        proximity_penalty = (MAX_LIDAR_DISTANCE - distance) / MAX_LIDAR_DISTANCE
                        lidar_penalty += proximity_penalty
                    # Reward for moving away from obstacles
                    else:
                        lidar_reward += LIDAR_REWARD_FACTOR

        if num_rays > 0:
            # Average the penalties and rewards over the number of rays
            average_lidar_penalty = lidar_penalty / num_rays
            average_lidar_reward = lidar_reward / num_rays

            total_lidar_penalty = average_lidar_penalty * LIDAR_PENALTY_FACTOR
            reward += total_lidar_penalty

            reward += average_lidar_reward  # Add the small reward for being far from obstacles

        # Check if the agent has picked up the ball
        if self.has_picked_up_ball():
            reward += PICKUP_REWARD
            terminated = True  # End the episode if ball is picked up

        # Check if out of bounds
        robot_pos = self.swerve_instances[0].get_box2d_instance().position
        if not (0 <= robot_pos.x <= 16.46 and 0 <= robot_pos.y <= 8.23):
            reward += OUT_OF_BOUNDS_PENALTY
            terminated = True

        return reward, terminated

    def get_camera_position(self, robot_position, robot_angle, camera_offset):
        x_offset, y_offset = camera_offset
        offset_x_rotated = x_offset * math.cos(robot_angle) - y_offset * math.sin(robot_angle)
        offset_y_rotated = x_offset * math.sin(robot_angle) + y_offset * math.cos(robot_angle)
        camera_x = robot_position[0] + offset_x_rotated
        camera_y = robot_position[1] + offset_y_rotated
        return camera_x, camera_y

    def close(self):
        pygame.quit()

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

        # Draw intake rectangle
        robot = self.swerve_instances[0].get_box2d_instance()
        robot_angle = robot.angle
        robot_position = robot.position

        # Intake rectangle corners in robot's local frame
        half_width = self.intake_width / 2
        half_height = self.intake_height / 2
        corners_local = [
            (self.intake_offset[0] - half_width, self.intake_offset[1] - half_height),
            (self.intake_offset[0] + half_width, self.intake_offset[1] - half_height),
            (self.intake_offset[0] + half_width, self.intake_offset[1] + half_height),
            (self.intake_offset[0] - half_width, self.intake_offset[1] + half_height)
        ]
        # Transform corners to world coordinates
        cos_angle = math.cos(robot_angle)
        sin_angle = math.sin(robot_angle)
        corners_world = []
        for x_local, y_local in corners_local:
            x_world = (x_local * cos_angle - y_local * sin_angle) + robot_position.x
            y_world = (x_local * sin_angle + y_local * cos_angle) + robot_position.y
            corners_world.append((x_world, y_world))
        # Convert to pygame coordinates
        corners_pygame = [self.CoordConverter.box2d_to_pygame(v) for v in corners_world]
        # Draw the intake rectangle
        pygame.draw.polygon(self.screen, (0, 255, 0), corners_pygame, 2)  # Green color, line width 2

        # Visualize depth estimations
        for camera_pos, look_at_pos, ray_intersections, cam_angle in self.depth_estimations:
            self.depth_estimator.visualize(
                self.screen, self.depth_estimator_boxes, self.depth_estimator_circles,
                camera_pos, look_at_pos, ray_intersections, self.CoordConverter
            )

        pygame.display.flip()
        self.clock.tick(self.TARGET_FPS)
