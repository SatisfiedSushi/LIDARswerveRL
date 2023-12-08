from FRCGameEnvSwerveLIDAR import env
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env=env()
env.reset()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

def animate(i, xs, ys):
    lidar_distances = env.step(0)[0].get('LIDAR Distances')
    lidar_angles = env.step(0)[0].get('LIDAR Angles')

    x_points =
