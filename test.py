import torch
from SimpleSwerveRLEnv import env
print(torch.cuda.is_available())

env=env()

obs, info = env.reset()
for i in range(100000000):
    obs, reward, terminated, truncated, info = env.step((0,0))
    print(reward)
    env.render()
    if truncated or terminated:
        obs, info = env.reset()
        print("Resetting")

env.close()