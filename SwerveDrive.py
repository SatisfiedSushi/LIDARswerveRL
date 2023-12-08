from typing import Optional, Union

import numpy as np
from Box2D.Box2D import b2Body


class SwerveDrive(object):
    def __init__(self, Box2d_instance: b2Body, team: Optional[str], angle: Optional[int or float], velocity: Optional[list or tuple], angular_velocity: Optional[int or float], velocity_factor: Optional[int or float], angular_velocity_factor: Optional[int or float]):
        self.Box2d_instance = Box2d_instance
        self.angle = angle
        self.team = team
        self.velocity_factor = velocity_factor
        self.angular_velocity_factor = angular_velocity_factor
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.velocity_with_factor = velocity * velocity_factor
        self.angular_velocity_with_factor = angular_velocity * angular_velocity_factor
        self.score = 0
        self.score_checked = False

    def get_score_checked(self) -> bool:
        return self.score_checked

    def get_score(self) -> int:
        self.score_checked = True
        return self.score

    def get_team(self) -> str:
        return self.team
    def get_position(self) -> dict:
        return {"x": self.Box2d_instance.position.x, "y": self.Box2d_instance.position.y}

    def get_box2d_instance(self) -> b2Body:
        return self.Box2d_instance

    def get_angle(self) -> Union[int, float]:
        return self.angle

    def get_velocity(self) -> Union[list or tuple]:
        return self.velocity

    def get_angular_velocity(self) -> Union[int, float]:
        return self.angular_velocity

    def get_velocity_with_factor(self) -> Union[list or tuple]:
        return self.velocity_with_factor

    def get_angular_velocity_with_factor(self) -> Union[int, float]:
        return self.angular_velocity_with_factor

    def set_score(self, score: int) -> None:
        self.score_checked = False
        self.score = score

    def set_team(self, team):
        self.team = team

    def set_angle(self, angle: Union[int, float]) -> None:
        self.angle = angle

    def set_velocity(self, velocity: Union[tuple]) -> None:
        self.velocity = velocity
        self.velocity_with_factor = (velocity[0] * self.velocity_factor, velocity[1] * self.velocity_factor)

    def set_angular_velocity(self, angular_velocity: Union[list]) -> None:
        self.angular_velocity = angular_velocity
        self.angular_velocity_with_factor = angular_velocity * self.angular_velocity_factor

    def update(self):

        vel = self.Box2d_instance.GetLinearVelocityFromWorldPoint(self.Box2d_instance.position)
        angular_vel = self.Box2d_instance.angularVelocity

        desired_x = 0
        desired_y = 0
        desired_theta = 0

        if self.get_velocity_with_factor()[0] < 0:
            desired_x = np.max([vel.x - 0.3, self.get_velocity_with_factor()[0]])
        elif self.get_velocity_with_factor()[0] == 0:
            desired_x = vel.x * 0.9
        elif self.get_velocity_with_factor()[0] > 0:
            desired_x = np.min([vel.x + 0.3, self.get_velocity_with_factor()[0]])

        if self.get_velocity_with_factor()[1] < 0:
            desired_y = np.max([vel.y - 0.3, self.get_velocity_with_factor()[1]])
        elif self.get_velocity_with_factor()[1] == 0:
            desired_y = vel.y * 0.9
        elif self.get_velocity_with_factor()[1] > 0:
            desired_y = np.min([vel.y + 0.3, self.get_velocity_with_factor()[1]])

        if self.get_angular_velocity_with_factor() < 0:
            desired_theta = np.max([angular_vel - 0.3, self.get_angular_velocity_with_factor()])
        elif self.get_angular_velocity_with_factor() == 0:
            desired_theta = angular_vel * 0.97
        elif self.get_angular_velocity_with_factor() > 0:
            desired_theta = np.min([angular_vel + 0.3, self.get_angular_velocity_with_factor()])

        vel_change_x = desired_x - vel.x
        vel_change_y = desired_y - vel.y
        vel_change_av = desired_theta - angular_vel

        impulse_x = self.Box2d_instance.mass * vel_change_x
        impulse_y = self.Box2d_instance.mass * vel_change_y
        impulse_av = vel_change_av

        max_impulse_av = 1.4

        if impulse_av > 0:
            impulse_av = np.min([impulse_av, max_impulse_av])
        elif impulse_av < 0:
            impulse_av = np.max([impulse_av, -max_impulse_av])

        '''closest_ball, offset_angle = return_closest_ball(body)'''
        '''if closest_ball is not None:
            print(f'LL.getXAngle = {offset_angle}')'''

        self.Box2d_instance.ApplyLinearImpulse((impulse_x, impulse_y), point=self.Box2d_instance.worldCenter, wake=True)
        self.Box2d_instance.ApplyAngularImpulse(impulse_av, wake=True)

        # main_robot.ApplyForce(force=(x, y), point=main_robot.__GetWorldCenter(), wake=True)

        angle_degrees = (self.Box2d_instance.angle / np.pi * 180)  # radians to degrees
        #self.set_angle(angle_degrees % 360)
        # print((angle_degrees % 360))  # converts to 0 to 360
