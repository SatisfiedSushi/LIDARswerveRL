from RayCastCallback import RayCastCallback
from Box2D.Box2D import *
import math


class LIDAR:
    def __init__(self, world):
        self.world = world

    def cast_rays(self, b2_robot, robot_position, robot_angle, robot_length, robot_width, number_of_rays, lidar_length):
        number_of_rays = 100
        lidar_length = 4
        # invert the robot angle
        robot_angle = -robot_angle
        # convert to radians below 2pi and above 0
        robot_angle = robot_angle
        ray_cast_callback = RayCastCallback()

        # define the angle between each ray
        angle_between_rays = 2 * math.pi / number_of_rays

        # define the starting angle
        starting_angle = robot_angle - math.pi

        # define the ending angle
        ending_angle = robot_angle + math.pi

        # define the current angle
        current_angle = starting_angle

        # define the distances of all LIDAR sensors
        distances = []

        ray_angles = []

        # ray end positions
        ray_end_positions = []

        converted_endpos = []

        raycast_points = []

        # cast a ray for each ray
        for ray_number in range(number_of_rays):
            # define the end position of the ray
            ray_end_position = b2Vec2(robot_position.x + (lidar_length / 2) * math.cos(current_angle),
                                      robot_position.y - (lidar_length / 2) * math.sin(current_angle))
            ray_end_positions.append((robot_position.x + (lidar_length / 2) * math.cos(current_angle),
                                      robot_position.y - (lidar_length / 2) * math.sin(current_angle)))

            ray_cast_callback = RayCastCallback()
            # cast the ray
            self.world.RayCast(ray_cast_callback, robot_position, ray_end_position)

            ray_angles.append(current_angle)

            # define the distance of the ray
            ray_distance = ray_cast_callback.m_fraction * lidar_length

            raycast_points.append(ray_cast_callback.m_point)

            # add the distance to the list of distances
            distances.append(ray_distance)

            # add the angle between rays to the current angle
            current_angle += angle_between_rays

        # return the distances of all LIDAR sensors
        return distances, ray_end_positions, ray_angles, converted_endpos, raycast_points
