import functools
import os
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

import math
import pygame

import pygame
import math


class RaycastVisualizer:
    def __init__(self, screen_width, screen_height, fov_angle=60, ray_count=100):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fov_angle = fov_angle
        self.ray_count = ray_count

    def rotate_point(self, x, y, cx, cy, angle):
        radians = math.radians(angle)
        cos = math.cos(radians)
        sin = math.sin(radians)
        nx = cos * (x - cx) - sin * (y - cy) + cx
        ny = sin * (x - cx) + cos * (y - cy) + cy

        return nx, ny

    def create_box(self, position, size, angle):
        """Create a box with size and rotation in the RaycastVisualizer's coordinate system."""
        print(f'box center: {position}')
        print(f'box size: {size}')
        cx, cy = position  # Box center in Pygame coordinates
        # convert to Box2D coordinates
        half_width, half_height = size / 2, size / 2

        # Define the corners before rotation
        corners = [
            (cx - half_width, cy - half_height),
            (cx + half_width, cy - half_height),
            (cx + half_width, cy + half_height),
            (cx - half_width, cy + half_height)
        ]

        # Rotate corners around the center of the box using the angle
        rotated_corners = [self.rotate_point(x, y, cx, cy, angle) for x, y in corners]
        print(f'rotated corners: {rotated_corners}')
        return rotated_corners


    def create_circle(self, center, radius):
        print(f'circle center: {center}')
        """Create a circle with a given radius."""
        # Simply return the center and radius, as a circle doesn't need rotation
        cx, cy = center
        return (cx, cy), radius  # Return center and radius, no rotation needed for circles

    def ray_intersects_segment(self, ray_origin, ray_direction, seg_start, seg_end):
        """Check for intersections between a ray and line segments."""
        ray_dx, ray_dy = ray_direction
        seg_dx, seg_dy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]

        denominator = (-seg_dx * ray_dy + ray_dx * seg_dy)
        if denominator == 0:
            return None  # Parallel lines

        t = (-ray_dy * (ray_origin[0] - seg_start[0]) + ray_dx * (ray_origin[1] - seg_start[1])) / denominator
        u = (seg_dx * (ray_origin[1] - seg_start[1]) - seg_dy * (ray_origin[0] - seg_start[0])) / denominator

        if 0 <= t <= 1 and u >= 0:
            intersection_x = seg_start[0] + t * seg_dx
            intersection_y = seg_start[1] + t * seg_dy
            return (intersection_x, intersection_y), u  # Return intersection and distance along the ray
        return None

    def cast_rays(self, camera_pos, look_at_pos, boxes, circles, robot_angle, max_ray_distance=500):
        """Raycasting from the robot's position, limiting the range of the rays."""
        cam_angle = self.calculate_angle(camera_pos, look_at_pos)
        half_fov = self.fov_angle / 2

        # Define the rays by their angles
        rays = []
        for i in range(self.ray_count):
            ray_angle = cam_angle - half_fov + i * (self.fov_angle / (self.ray_count - 1))
            adjusted_ray_angle = ray_angle + robot_angle  # Adjust with robot's current angle
            ray_direction = (math.cos(math.radians(adjusted_ray_angle)), math.sin(math.radians(adjusted_ray_angle)))
            rays.append((adjusted_ray_angle, ray_direction))

        ray_intersections = []
        for ray_angle, ray_direction in rays:
            closest_intersection = None
            closest_distance = max_ray_distance  # Limit ray range
            hit_object_id = None

            # Check intersection with boxes
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

            # Check intersection with circles
            for circle_id, (center, radius) in enumerate(circles):
                intersection = self.ray_intersects_circle(camera_pos, ray_direction, center, radius)
                if intersection:
                    point, distance = intersection
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_intersection = point
                        hit_object_id = circle_id + len(boxes)

            ray_intersections.append((closest_intersection, closest_distance, hit_object_id))

        return ray_intersections

    def ray_intersects_circle(self, ray_origin, ray_direction, circle_center, circle_radius):
        """Check for intersection between a ray and a circle."""
        ox, oy = ray_origin
        dx, dy = ray_direction
        cx, cy = circle_center
        r = circle_radius

        # Vector from ray origin to circle center
        oc_x = cx - ox
        oc_y = cy - oy

        # Project vector onto ray direction to get closest approach
        t_closest = (oc_x * dx + oc_y * dy) / (dx * dx + dy * dy)
        closest_x = ox + t_closest * dx
        closest_y = oy + t_closest * dy

        # Distance from closest approach to circle center
        dist_to_center = math.hypot(closest_x - cx, closest_y - cy)

        if dist_to_center > r:
            return None  # No intersection

        # Distance from closest approach to intersection point
        t_offset = math.sqrt(r ** 2 - dist_to_center ** 2)

        # Find intersection points along the ray
        t1 = t_closest - t_offset
        t2 = t_closest + t_offset

        # We want the closest positive intersection along the ray
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
        """Calculate the angle in degrees between two points, with proper conversion for coordinate systems."""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]  # No need to invert the y-axis for angle calculation
        angle = math.degrees(math.atan2(dy, dx))

        # Adjust the angle based on coordinate system differences
        # Pygame's angles increase clockwise, so we invert the angle to match Box2D
        angle = -angle  # This corrects the inversion

        return angle

    def draw_2d_perspective(self, screen, boxes, circles, camera_pos, look_at_pos, robot_angle, agent_id=None):
        """Draw the 2D perspective with raycasting and FOV visualization, skipping the agent itself."""
        # Draw boxes
        for box in boxes:
            pygame.draw.polygon(screen, (0, 0, 0), box, 2)

        # Draw camera and look-at point
        pygame.draw.circle(screen, (255, 0, 0), camera_pos, 5)
        pygame.draw.circle(screen, (0, 255, 0), look_at_pos, 5)

        # Draw FOV lines
        cam_angle = self.calculate_angle(camera_pos, look_at_pos)
        half_fov = self.fov_angle / 2
        fov_left_angle = cam_angle - half_fov
        fov_right_angle = cam_angle + half_fov
        fov_length = 500

        left_x = camera_pos[0] + fov_length * math.cos(math.radians(fov_left_angle))
        left_y = camera_pos[1] + fov_length * math.sin(math.radians(fov_left_angle))
        right_x = camera_pos[0] + fov_length * math.cos(math.radians(fov_right_angle))
        right_y = camera_pos[1] + fov_length * math.sin(math.radians(fov_right_angle))

        pygame.draw.line(screen, (128, 128, 128), camera_pos, (left_x, left_y), 1)
        pygame.draw.line(screen, (128, 128, 128), camera_pos, (right_x, right_y), 1)

        # Draw rays
        ray_intersections = self.cast_rays(camera_pos, look_at_pos, boxes, circles, robot_angle)
        for intersection, _, _ in ray_intersections:
            if intersection:
                pygame.draw.line(screen, (255, 0, 0), camera_pos, intersection, 1)

        return ray_intersections

    def print_1d_perspective(self, ray_intersections):
        """Print the 1D perspective with angles."""
        object_perspective = {}
        prev_object_id = None
        start_angle = None

        half_fov = self.fov_angle / 2

        for i, (_, _, object_id) in enumerate(ray_intersections):
            # Normalize the ray_angle from [-half_fov, half_fov] to [-1, 1]
            ray_angle = (-half_fov + i * (self.fov_angle / (self.ray_count - 1))) / half_fov

            if object_id != prev_object_id:
                if prev_object_id is not None:
                    # Store the previous object with its start and end angles
                    object_name = f"obj{prev_object_id + 1}"
                    object_perspective[object_name] = (start_angle, ray_angle)
                start_angle = ray_angle  # New object's start angle

            prev_object_id = object_id

        if prev_object_id is not None:
            # Store the last object
            object_name = f"obj{prev_object_id + 1}"
            object_perspective[object_name] = (start_angle, ray_angle)

        # print(object_perspective)

    def draw_1d_perspective_with_gaps(self, screen, ray_intersections):
        """Draw the 1D perspective with gap detection."""
        # Draw origin at camera in 1D space (camera's 1D projection)
        pygame.draw.circle(screen, (255, 0, 0), (300, 100), 5)

        # Scale factor for distance normalization
        max_distance = max([distance for _, distance, _ in ray_intersections if distance != float('inf')])
        if max_distance == 0:
            max_distance = 1  # Prevent division by zero

        # Plot each intersection along the x-axis of the 1D canvas based on angular position
        half_canvas_width = 300
        half_fov = self.fov_angle / 2
        prev_x = None
        gap_threshold = 10  # Define a gap threshold for detecting significant distance changes
        prev_object_id = None  # Track the previous object that the ray hit

        for i, (_, distance, object_id) in enumerate(ray_intersections):
            if distance != float('inf'):
                # Calculate the angular position relative to the center of the FOV
                ray_angle = (-half_fov + i * (self.fov_angle / (self.ray_count - 1)))  # Angular position
                normalized_angle = ray_angle / self.fov_angle  # Normalize angle to [-0.5, 0.5]

                # Project based on angular position, centering around the canvas
                proj_x = 300 + normalized_angle * half_canvas_width

                if prev_x is not None:
                    if abs(distance - prev_distance) < gap_threshold and object_id == prev_object_id:
                        # Connect the previous point to the current point if no gap is detected
                        pygame.draw.line(screen, (0, 0, 255), (prev_x, 100), (proj_x, 100), 1)

                prev_x = proj_x
                prev_distance = distance
                prev_object_id = object_id  # Update the last object hit



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
        x_box2d = x / self.PPM
        y_box2d = (self.SCREEN_HEIGHT - y) / self.PPM  # Invert y-coordinate

        # Debugging output
        print(f"Pygame -> Box2D: ({x}, {y}) -> ({x_box2d}, {y_box2d})")

        return x_box2d, y_box2d

    def box2d_to_pygame(self, position_box2d):
        x_box2d, y_box2d = position_box2d
        x_pygame = x_box2d * self.PPM
        y_pygame = self.SCREEN_HEIGHT - (y_box2d * self.PPM)  # Y-axis flip for Pygame

        # Debugging output
        print(f"Box2D -> Pygame: ({x_box2d}, {y_box2d}) -> ({x_pygame}, {y_pygame})")

        return (x_pygame, y_pygame)

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
        # new_ball.ApplyLinearImpulse((np.cos(force_direction) * force, np.sin(force_direction) * force),
        #                             point=new_ball.worldCenter, wake=True)

    def create_random_ball(self):
        # Adjusted ranges to account for a 1 meter border
        x_range = (1, 15.46)  # 1 meter away from both left and right edges
        y_range_top = (1, 2)  # Top 2 meters but 1 meter away from top edge
        y_range_bottom = (6.23, 7.23)  # Bottom 2 meters but 1 meter away from bottom edge

        # Randomly choose to place the ball at the top or bottom
        x_position = random.uniform(*x_range)
        if random.choice([True, False]):
            y_position = random.uniform(*y_range_top)
        else:
            y_position = random.uniform(*y_range_bottom)

        # Call existing method to create a ball at the selected position
        self.create_new_ball(position=(x_position, y_position), force_direction=0,
                             team=random.choice(["Red", "Blue"]))

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
        robot_angle = robot.angle
        robot_angle = math.degrees(robot_angle)
        robot_angle = self.normalize_angle_degrees(robot_angle)
        ball_position = ball.position
        robot_position = robot.position

        angle_to_ball = math.degrees(math.atan2(ball_position.y - robot_position.y, ball_position.x - robot_position.x)) + 30
        angle_to_ball = self.normalize_angle_degrees(angle_to_ball)

        relative_angle = self.normalize_angle_degrees(angle_to_ball - robot_angle)

        # print(f"Robot angle: {robot_angle}, Angle to ball: {angle_to_ball}, Relative angle: {relative_angle}")

        # Assuming the intake side is at the front, check if the relative angle is within a certain threshold
        threshold_angle = 70  # Example: 30 degrees
        return abs(relative_angle) < threshold_angle

    def has_picked_up_ball(self, ball):
        robot = self.swerve_instances[0].get_box2d_instance()
        ball_position = ball.position
        robot_position = robot.position

        # Check if the ball is close enough to the robot's intake side
        distance = math.hypot(ball_position.x - robot_position.x, ball_position.y - robot_position.y)
        is_aligned = self.is_robot_aligned_with_ball(ball)

        # Define a threshold for how close the ball needs to be to consider it picked up
        pickup_distance_threshold = 0.8  # Example: Increase the threshold to 1.0 units
        # return distance < pickup_distance_threshold and is_aligned
        return False

    def normalize_angle_degrees(self, angle):
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle

    def calculate_angle_to_ball(self, ball):
        if ball is None:
            return 0  # Return a default value if there's no ball

        robot = self.swerve_instances[0].get_box2d_instance()
        robot_position = robot.position
        ball_position = ball.position

        # Calculate the angle from the robot to the ball
        delta_x = ball_position.x - robot_position.x
        delta_y = ball_position.y - robot_position.y
        angle_to_ball = math.atan2(delta_y, delta_x)

        # Adjust the angle based on the robot's current orientation
        robot_angle = robot.angle
        robot_angle = math.degrees(robot_angle)
        robot_angle = self.normalize_angle_degrees(robot_angle)
        angle_relative_to_robot = angle_to_ball - robot_angle

        # Normalize the angle to the range [-pi, pi]
        angle_relative_to_robot = (angle_relative_to_robot + math.pi) % (2 * math.pi) - math.pi

        return angle_relative_to_robot

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

    def get_middle_line_distances(self):
        """
        Simulate getting distance values for each pixel along the middle horizontal line in the environment.
        Gradual changes in distances to simulate smoother transitions.
        """
        middle_y = self.SCREEN_HEIGHT // 2
        distances = []
        last_distance = random.uniform(5, 10)  # Start with a base distance

        for x in range(self.SCREEN_WIDTH):
            # Smaller, more controlled gradual changes for smoother transitions
            change = random.uniform(-0.05, 0.05)
            simulated_distance = max(1, min(10, last_distance + change))  # Ensure values stay within a reasonable range
            distances.append(simulated_distance)
            last_distance = simulated_distance

        return distances

    def get_camera_position(self, robot_position, robot_angle, camera_offset):
        """Get the camera position with an adjustable offset relative to the robot."""
        x_offset, y_offset = camera_offset

        # Rotate the offset based on the robot's angle
        offset_x_rotated = x_offset * math.cos(robot_angle) - y_offset * math.sin(robot_angle)
        offset_y_rotated = x_offset * math.sin(robot_angle) + y_offset * math.cos(robot_angle)

        camera_x = robot_position[0] + offset_x_rotated
        camera_y = robot_position[1] + offset_y_rotated

        return camera_x, camera_y

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

    def __init__(self, render_mode="human", max_teleop_time=5):
        super().__init__()


        self.LIDAR_active = False
        # --- pygame setup ---
        # self.end_goal = ((random.randint(200, 1446) / 100), (random.randint(200, 623) / 100))
        # self.end_goal = (5, 7)
        self.PPM = 100.0  # pixels per meter
        self.TARGET_FPS = 60
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self.meters_to_pixels(16.46), self.meters_to_pixels(8.23)
        self.screen = None
        self.clock = None
        self.teleop_time = max_teleop_time  # 135 default
        self.CoordConverter = CoordConverter()
        self.starting_balls = 1
        self.balls = []


        # RL variables
        self.render_mode = render_mode
        # self.possible_agents = ["blue_1", "blue_2", "blue_3", "red_1", "red_2", "red_3"]
        # self.agent_ids = ["blue_1", "blue_2", "blue_3", "red_1", "red_2", "red_3"]
        self.possible_agents = ["blue_1"]
        self.agent_ids = ["blue_1"]
        self.agents = copy(self.possible_agents)
        self.resetted = False
        self.raycast_visualizer = RaycastVisualizer(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, fov_angle=60, ray_count=100)

        self.previous_angle_to_ball = 0

        self.number_of_rays = 400
        # the end ray positions
        self.end_rays = []
        self.ray_distances = []
        self.ray_angles = []
        self.b2LIDARs = []
        self.distance_endpoints = []
        self.raycast_points = []
        # self.observation_space = Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]), high=np.array([16.46, 8.23, 360, 16.46, 8.23]), shape=(5,))
        self.observation_space = Box(
            low=
                np.array([
                    -np.inf, -np.inf,  # robot position x, y
                    -np.inf,  # robot angle
                    -np.inf, -np.inf,  # robot velocity x, y
                    0,  # distance to ball
                    -np.pi,  # angle to ball
                    0,  # ball picked up indicator
                    0,  # game time remaining,
                    0   # average LIDAR distance
                ]),
            high=
                np.array([
                    np.inf, np.inf,  # robot position x, y
                    np.inf,  # robot angle
                    np.inf, np.inf,  # robot velocity x, y
                    np.inf,  # distance to ball
                    np.pi,  # angle to ball
                    1,  # ball picked up indicator
                    max_teleop_time,  # game time remaining
                    np.inf  # average LIDAR distance
                ])
        )
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

        self.previous_distance_to_ball = 0

        self.timestep = None
        self.current_time = None
        self.distances = None
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
        self.obstacles = None
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
            # Convert each vertex from Box2D to Pygame coordinates
            vertices = [body.GetWorldPoint(v) for v in polygon.vertices]  # Properly transform each vertex to world coordinates

            vertices = [self.box2d_to_pygame(v) for v in vertices]

            # Debugging output for translated vertices
            print(f'Translated Vertices: {vertices}')

            if body.userData is not None and 'Team' in body.userData:
                # Use different colors based on the 'Team' key if it exists
                color = (128, 128, 128, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255)
            else:
                # Default color for bodies without 'Team' key
                color = (255, 255, 255, 255)

            pygame.draw.polygon(self.screen, color, vertices)

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

    def create_obstacle(self, position, size, angle=0):
        """
        Create a rectangular static obstacle with no friction, given a position, size (width, height), and rotation angle.
        """
        x, y = position
        print(f"Creating obstacle at position ({x}, {y}) with size {size} and angle {angle} degrees")

        # Create the obstacle in the Box2D world as a static body
        obstacle = self.world.CreateStaticBody(
            position=(x, y),
            angle=math.radians(angle),
            userData={"obstacle": True, "isFlaggedForDelete": False}  # Optional: user data can be added as needed
        )

        # Create a rectangular fixture for the obstacle
        obstacle.CreatePolygonFixture(
            box=(size[0] / 2, size[1] / 2),  # Define the box using half the width and height
            density=0,  # Static objects should have zero density
            friction=0  # No friction for the static obstacle
        )

        return obstacle

    def rotate_point(self, x, y, cx, cy, angle):
        """Rotate a point (x, y) around a center point (cx, cy) by a given angle (in degrees)."""
        radians = math.radians(angle)
        cos = math.cos(radians)
        sin = math.sin(radians)
        nx = cos * (x - cx) - sin * (y - cy) + cx
        ny = sin * (x - cx) + cos * (y - cy) + cy
        return nx, ny

    def reset(self, *, seed=None, options=None):

        # --- RL variables ---
        self.distances = []
        self.previous_angle_to_ball = 0
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.scoreHolder = ScoreHolder()
        self.current_time = time.time()
        self.game_time = self.teleop_time - (time.time() - self.current_time)
        self.CoordConverter = CoordConverter()
        self.previous_distance_to_ball = 0
        # self.end_goal = (np.random.randint(140, 1500) / 100, np.random.randint(140, 750) / 100)
        # self.end_goal = (5, 7)

        # --- other ---
        self.balls = []
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



        if self.LIDAR_active:
            self.sample_object = self.world.CreateStaticBody(
                position=(16.47/2, 8.23/2),
                shapes=b2PolygonShape(box=(1, 1)),
            )



            self.obstacles = [
                self.create_obstacle((4, 3), (0.5, 0.5)),
                self.create_obstacle((8, 4), (0.5, 0.5))
            ]

        self.obstacles = [
            self.create_obstacle((4, 3), (0.5, 0.5)),
            self.create_obstacle((6, 2), (1, 1))
        ]

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
            self.robots]

        self.scoreHolder.set_swerves(swerves=self.swerve_instances)

        for i in range(self.starting_balls):
            self.create_random_ball()

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
        obs = np.zeros(self.observation_space.shape)

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
        ALIGNMENT_REWARD = 0.4
        PICKUP_REWARD = 200.0
        OUT_OF_BOUNDS_PENALTY = -50.0
        LIDAR_DISTANCE_THRESHOLD = 2.0  # Adjust this value as needed
        LIDAR_REWARD_SCALING_FACTOR = 5.0
        ANGLE_REWARD = 0.2  # New reward for reducing the angle to the ball

        terminated = False
        log_messages = []  # List to accumulate log messages

        # Find the closest ball
        closest_ball, distance_to_ball = self.find_closest_ball()

        # Initialize reward with time step penalty
        reward = TIME_STEP_PENALTY
        log_messages.append(f"Time step penalty applied: {TIME_STEP_PENALTY}")

        # Reward for getting closer to the ball
        if distance_to_ball < self.previous_distance_to_ball:
            reward += PROGRESS_REWARD
            log_messages.append(f"Progress reward added: {PROGRESS_REWARD}")

        # Reward for aligning intake side towards the ball
        if self.is_robot_aligned_with_ball(closest_ball):
            reward += ALIGNMENT_REWARD
            log_messages.append(f"Alignment reward added: {ALIGNMENT_REWARD}")

        # Reward for reducing the angle to the ball
        angle_to_ball = self.calculate_angle_to_ball(closest_ball)
        if angle_to_ball < self.previous_angle_to_ball:
            reward += ANGLE_REWARD
            log_messages.append(f"Angle reward added: {ANGLE_REWARD}")

        # Reward for picking up the ball
        if self.has_picked_up_ball(closest_ball):
            reward += PICKUP_REWARD
            terminated = True
            log_messages.append(f"Ball pickup reward added: {PICKUP_REWARD}")

        # Penalty for going out of bounds
        robot_pos = self.swerve_instances[0].get_box2d_instance().position
        if not (0 <= robot_pos.x <= 16.46 and 0 <= robot_pos.y <= 8.23):
            reward += OUT_OF_BOUNDS_PENALTY
            terminated = True
            log_messages.append(f"Out of bounds penalty applied: {OUT_OF_BOUNDS_PENALTY}")

        # Calculate the average LIDAR distance
        if self.LIDAR_active:
            if self.distances:  # Check if self.distances is not empty
                filtered_distances = [distance ** 2 for distance in self.distances if distance != 0]
                if filtered_distances:  # Check if there are non-zero distances
                    average_lidar_distance = sum(filtered_distances) / len(filtered_distances)
                else:
                    average_lidar_distance = 0  # Handle case where all distances are 0
            else:
                average_lidar_distance = 0  # Handle case where self.distances is empty

            # Add penalty based on the average LIDAR distance
            reward -= LIDAR_REWARD_SCALING_FACTOR / average_lidar_distance if average_lidar_distance else 0
            log_messages.append(f"LIDAR penalty applied: {LIDAR_REWARD_SCALING_FACTOR / average_lidar_distance if average_lidar_distance else 0}")

        # Update previous distance and angle for the next calculation
        self.previous_distance_to_ball = distance_to_ball
        self.previous_angle_to_ball = angle_to_ball

        # Print the consolidated log messages
        # for message in log_messages:
        #     if "LIDAR" in message:
        #         print(message)

        return reward, terminated

    def step(self, actions, testing_mode=False):  # TODO: change action dictionary
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
        if testing_mode:
            swerve.set_velocity((self.D - self.A, self.W - self.S))
            swerve.set_angular_velocity(self.LEFT - self.RIGHT)
        else:
            swerve.set_velocity((actions[0], actions[1]))
            # swerve.set_velocity((0,0))
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

        if self.LIDAR_active:
            # Cast LIDAR rays
            self.distances, self.ray_end_positions, self.ray_angles, self.converted_endpos, self.raycast_points = self.LIDAR.cast_rays(
                swerve.get_box2d_instance(),
                swerve.get_box2d_instance().position,
                swerve.get_angle(),
                0.56,  # Assuming the robot length is 0.56
                0.56,  # Assuming the robot width is 0.56
                100,  # Number of rays
                4  # LIDAR length
            )

        # Find the closest ball at the beginning of each step
        closest_ball, distance_to_ball = self.find_closest_ball()

        # Calculate angle to the closest ball
        angle_to_ball = self.calculate_angle_to_ball(closest_ball)

        # obs = (self.swerve_instances[0].get_box2d_instance().position.x,
        #        self.swerve_instances[0].get_box2d_instance().position.y,
        #        self.swerve_instances[0].get_angle(),
        #        self.end_goal[0],
        #        self.end_goal[1])

        robot = self.swerve_instances[0].get_box2d_instance()
        velocity_x, velocity_y = self.swerve_instances[0].get_velocity()
        average_lidar_distance = 0
        if self.LIDAR_active:
            if self.distances:  # Check if self.distances is not empty
                filtered_distances = [distance ** 2 for distance in self.distances if distance != 0]
                if filtered_distances:  # Check if there are non-zero distances
                    average_lidar_distance = sum(filtered_distances) / len(filtered_distances)
                else:
                    average_lidar_distance = 0  # Handle case where all distances are 0
            else:
                average_lidar_distance = 0  # Handle case where self.distances is empty

        # Convert Box2D objects into visualizer-friendly boxes
        boxes = []
        # Example obstacle rendering
        for obstacle in self.obstacles:
            # Retrieve the position and angle of the obstacle from Box2D
            position_box2d = obstacle.position  # Box2D position (center of the obstacle)
            angle = obstacle.angle  # Box2D angle

            # Get the vertices from the obstacle's fixture
            vertices_box2d = [obstacle.transform * v for v in obstacle.fixtures[0].shape.vertices]

            # Rotate the vertices based on the obstacle's angle and calculate their positions
            rotated_vertices_box2d = [
                self.raycast_visualizer.rotate_point(v[0], v[1], position_box2d[0], position_box2d[1],
                                                     math.degrees(angle))
                for v in vertices_box2d
            ]

            # Calculate the min and max x and y values from the rotated vertices
            min_x = min(v[0] for v in rotated_vertices_box2d)
            max_x = max(v[0] for v in rotated_vertices_box2d)
            min_y = min(v[1] for v in rotated_vertices_box2d)
            max_y = max(v[1] for v in rotated_vertices_box2d)

            # Calculate the dynamic width and height based on the difference between the min/max x and y values
            dynamic_width = max_x - min_x
            dynamic_height = max_y - min_y

            # Convert the center of the bounding box from Box2D to Pygame space (center of the obstacle)
            position_pygame = self.CoordConverter.box2d_to_pygame(((min_x + max_x) / 2, (min_y + max_y) / 2))

            # Convert the rotated vertices to Pygame space for rendering
            vertices_pygame = [self.CoordConverter.box2d_to_pygame(v) for v in rotated_vertices_box2d]

            # Log or print for debugging purposes
            print(f"Dynamic size for obstacle: Width={dynamic_width}, Height={dynamic_height}")
            print(f"Rotated Vertices (Pygame coordinates): {vertices_pygame}")

            # Add the transformed Pygame vertices to the list for rendering
            boxes.append(vertices_pygame)

        # Include other Box2D objects like balls and robots...

        # Get camera and look_at position for the agent's view
        camera_position = self.CoordConverter.box2d_to_pygame(self.swerve_instances[0].get_box2d_instance().position)
        look_at_position = (
        camera_position[0] + math.cos(robot.angle) * 100, camera_position[1] + math.sin(robot.angle) * 100)

        # Cast rays and draw visualization
        robot_angle = robot.angle  # Assuming you already have the robot angle calculated
        # Assuming you have a list of circles, possibly for balls or other round objects in your simulation.
        circles = []  # If there are no circles to pass, you can use an empty list.

        # Updated call
        ray_intersections = self.raycast_visualizer.cast_rays(camera_position, look_at_position, boxes, circles,
                                                              robot_angle)

        self.raycast_visualizer.draw_2d_perspective(self.screen, boxes, [], camera_position, look_at_position, robot_angle)


        obs = np.array([
            robot.position.x,
            robot.position.y,
            self.swerve_instances[0].get_angle(),
            velocity_x,
            velocity_y,
            distance_to_ball,
            angle_to_ball,
            int(self.has_picked_up_ball(closest_ball)),
            self.game_time,
            average_lidar_distance
        ])

        info = {}

        # Find the closest ball
        closest_ball, distance_to_ball = self.find_closest_ball()

        # Check if ball is picked up and robot is aligned
        if self.has_picked_up_ball(closest_ball) and self.is_robot_aligned_with_ball(closest_ball):
            terminated = True

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

        # render sample object
        if self.LIDAR_active:
            for fixture in self.sample_object.fixtures:
                fixture.shape.draw(self.sample_object, fixture)

        swerve = self.swerve_instances[0]

        for ball in self.balls:
            # adjusted ball position where the robot is centered at (0, 0)
            for fixture in ball.fixtures:
                fixture.shape.draw(ball, fixture)
            print(f'box2d ball position from actual fixture: {ball.position}')
        # render LIDAR rays

        for agent in self.agents:
            swerve = self.swerve_instances[self.agents.index(agent)]
            for fixture in swerve.get_box2d_instance().fixtures:
                fixture.shape.draw(swerve.get_box2d_instance(), fixture)

        max_distance = 4  # The length of the LIDAR rays
        distance_ratios = [distance / max_distance for distance in self.distances]

        for obstacle in self.obstacles:
            for fixture in obstacle.fixtures:
                fixture.shape.draw(obstacle, fixture)
            print(f'box2d obstacle position from actual fixture: {obstacle.position}')

        # for obstacle in self.obstacles:
        #     position_box2d = obstacle.position
        #     position_pygame = self.box2d_to_pygame(position_box2d)
        #
        #     # Debugging output
        #     print(f"Obstacle Position Box2D: {position_box2d} -> Pygame: {position_pygame}")
        #
        #     size = obstacle.fixtures[0].shape.radius * self.PPM  # Assuming a circular obstacle
        #     pygame.draw.circle(self.screen, (255, 0, 0), position_pygame, int(size))



        # Inside your render method:


        robot = self.swerve_instances[0].get_box2d_instance()
        robot_pos = robot.position
        robot_angle = robot.angle


        # Use the DepthVisualizer to display the Limelight's middle line with object widths

        FOV_angle = 30  # Half of the 60-degree FOV
        FOV_length = 4  # The length of the FOV lines

        # Calculate the FOV line endpoints
        left_line_angle = robot_angle + math.radians(FOV_angle)
        right_line_angle = robot_angle - math.radians(FOV_angle)

        # Left line
        left_line_end = (
            robot_pos.x + FOV_length * math.cos(left_line_angle),
            robot_pos.y + FOV_length * math.sin(left_line_angle)
        )
        pygame.draw.line(self.screen, (0, 255, 0),
                         self.CoordConverter.box2d_to_pygame(robot_pos),
                         self.CoordConverter.box2d_to_pygame(left_line_end), 2)

        # Right line
        right_line_end = (
            robot_pos.x + FOV_length * math.cos(right_line_angle),
            robot_pos.y + FOV_length * math.sin(right_line_angle)
        )
        pygame.draw.line(self.screen, (0, 255, 0),
                         self.CoordConverter.box2d_to_pygame(robot_pos),
                         self.CoordConverter.box2d_to_pygame(right_line_end), 2)

        if self.LIDAR_active:


            for end_position, distance_ratio in zip(self.ray_end_positions, distance_ratios):
                # Interpolate between green and red based on the distance ratio
                # Reverse the ratio for the red component to make it more red when closer
                # Keep the green component high when the object is far and decrease it as the object gets closer
                color = (int(255 * (distance_ratio)), int(255 * (1 - distance_ratio)), 0)
                pygame.draw.line(
                    self.screen, color,
                    self.CoordConverter.box2d_to_pygame(swerve.get_box2d_instance().position),
                    self.CoordConverter.box2d_to_pygame(end_position)
                )

        # Draw raycasts and FOV
        camera_pos = self.box2d_to_pygame(self.swerve_instances[0].get_box2d_instance().position)
        look_at_pos = (camera_pos[0] + math.cos(self.swerve_instances[0].get_box2d_instance().angle) * 100,
                       camera_pos[1] + math.sin(self.swerve_instances[0].get_box2d_instance().angle) * 100)
        robot_angle = self.swerve_instances[0].get_box2d_instance().angle

        # Convert Box2D objects into visualizer-friendly boxes
        boxes = []
        circles = []

        # Add obstacles as boxes
        for obstacle in self.obstacles:
            # Retrieve the position and angle of the obstacle from Box2D
            position_box2d = obstacle.position  # Box2D position (center of the obstacle)
            angle = obstacle.angle  # Box2D angle

            # Get the vertices from the obstacle's fixture
            vertices_box2d = [obstacle.transform * v for v in obstacle.fixtures[0].shape.vertices]

            # Rotate the vertices based on the obstacle's angle and calculate their positions
            rotated_vertices_box2d = [
                self.raycast_visualizer.rotate_point(v[0], v[1], position_box2d[0], position_box2d[1],
                                                     math.degrees(angle))
                for v in vertices_box2d
            ]

            # Calculate the min and max x and y values from the rotated vertices
            min_x = min(v[0] for v in rotated_vertices_box2d)
            max_x = max(v[0] for v in rotated_vertices_box2d)
            min_y = min(v[1] for v in rotated_vertices_box2d)
            max_y = max(v[1] for v in rotated_vertices_box2d)

            # Calculate the dynamic width and height based on the difference between the min/max x and y values
            dynamic_width = max_x - min_x
            dynamic_height = max_y - min_y

            # Convert the center of the bounding box from Box2D to Pygame space (center of the obstacle)
            position_pygame = self.CoordConverter.box2d_to_pygame(((min_x + max_x) / 2, (min_y + max_y) / 2))

            # Convert the rotated vertices to Pygame space for rendering
            vertices_pygame = [self.CoordConverter.box2d_to_pygame(v) for v in rotated_vertices_box2d]

            # Log or print for debugging purposes
            print(f"Dynamic size for obstacle: Width={dynamic_width}, Height={dynamic_height}")
            print(f"Rotated Vertices (Pygame coordinates): {vertices_pygame}")

            # Add the transformed Pygame vertices to the list for rendering
            boxes.append(vertices_pygame)

        # Debug: Draw bounding boxes for obstacles
        for obstacle in self.obstacles:
            for fixture in obstacle.fixtures:
                if isinstance(fixture.shape, b2PolygonShape):
                    # Get the vertices of the polygon and draw them
                    vertices = [(obstacle.transform * v) * self.PPM for v in fixture.shape.vertices]
                    vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]  # Adjust coordinates for pygame
                    pygame.draw.polygon(self.screen, (0, 255, 0), vertices, 2)  # Draw obstacle hitbox in green

        # Convert balls to circles
        for ball in self.balls:
            print(f'box2d ball position: {ball.position}')
            position = self.CoordConverter.box2d_to_pygame(ball.position)
            radius = ball.fixtures[0].shape.radius * self.PPM
            circle = self.raycast_visualizer.create_circle(position, radius)
            circles.append(circle)

        for box in boxes:
            print(f"Drawing box with corners: {box}")
            pygame.draw.polygon(self.screen, (0, 255, 0), box, 2)

        for idx, agent in enumerate(self.robots):
            if agent == self.swerve_instances[0].get_box2d_instance():
                continue  # Skip adding the agent's own bounding box for raycasting

            position = self.CoordConverter.box2d_to_pygame(agent.position)
            angle = math.degrees(agent.angle)
            size = 0.56 * self.PPM  # Assuming robot size
            box = self.raycast_visualizer.create_box(position, size, angle)
            boxes.append(box)
        # Draw the perspective and raycasts, skipping the agent's own box
        ray_intersections = self.raycast_visualizer.draw_2d_perspective(
            self.screen, boxes, circles, camera_pos, look_at_pos, robot_angle
        )

        camera_offset = (0.25, 0)  # (x_offset, y_offset)
        camera_x, camera_y = self.get_camera_position(self.swerve_instances[0].get_box2d_instance().position,
                                                      robot_angle, camera_offset)
        camera_position = self.CoordConverter.box2d_to_pygame((camera_x, camera_y))

        # Set the look_at_position to be in the robot's forward direction
        look_at_position = (
            camera_position[0] + math.cos(robot_angle) * 100,  # Use robot_angle to point forward
            camera_position[1] + math.sin(robot_angle) * 100
        )

        # Draw 2D perspective using the RaycastVisualizer
        ray_intersections = self.raycast_visualizer.cast_rays(camera_position, look_at_position, boxes, circles, robot_angle)


        # Optionally, print the 1D perspective from raycasting results
        self.raycast_visualizer.print_1d_perspective(ray_intersections)

        # Debug: Draw raycast hit points on obstacles
        for intersection, _, _ in ray_intersections:
            if intersection:
                pygame.draw.circle(self.screen, (255, 0, 0), intersection, 3)  # Draw red dot at intersection point

        # Define the distance of the FOV from the robot and its angle
        FOV_distance = 0.8  # This is the pickup_distance_threshold used in has_picked_up_ball function
        FOV_angle = 70  # This is the threshold_angle used in is_robot_aligned_with_ball function

        # Calculate the start and end points of the inner and outer lines of the FOV
        inner_line_start = robot_pos
        inner_line_end = (robot_pos[0] - FOV_distance * math.cos(robot_angle - FOV_angle / 2),
                          robot_pos[1] - FOV_distance * math.sin(robot_angle - FOV_angle / 2))

        outer_line_start = robot_pos
        outer_line_end = (robot_pos[0] - FOV_distance * math.cos(robot_angle + FOV_angle / 2),
                          robot_pos[1] - FOV_distance * math.sin(robot_angle + FOV_angle / 2))

        # Convert the start and end points from Box2D coordinates to Pygame coordinates
        inner_line_start = self.CoordConverter.box2d_to_pygame(inner_line_start)
        inner_line_end = self.CoordConverter.box2d_to_pygame(inner_line_end)
        outer_line_start = self.CoordConverter.box2d_to_pygame(outer_line_start)
        outer_line_end = self.CoordConverter.box2d_to_pygame(outer_line_end)

        # Draw the inner and outer lines of the FOV
        pygame.draw.line(self.screen, (255, 0, 0), inner_line_start, inner_line_end)
        pygame.draw.line(self.screen, (255, 0, 0), outer_line_start, outer_line_end)

        pygame.draw.circle(self.screen, (0, 0, 255), camera_position, 5)  # Blue dot for the camera position

        for intersection, _, _ in ray_intersections:
            if intersection:
                pygame.draw.circle(self.screen, (255, 0, 0), intersection, 3)  # Draw red dot at intersection point

        # print cumulative distance
        # print(f'cumulative distance: {sum([distance**2 for distance in self.distances])}')

        # pygame.draw.circle(self.screen, ('red'), self.CoordConverter.box2d_to_pygame(self.end_goal), 7)

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
