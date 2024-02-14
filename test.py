import torch
from SimpleSwerveRLEnvIntake import env
print(torch.cuda.is_available())

env=env(max_teleop_time=5000)

obs, info = env.reset()
for i in range(100000000):
    obs, reward, terminated, truncated, info = env.step((0,0), testing_mode=True)
    # print(reward)
    env.render()
    if truncated or terminated:
        obs, info = env.reset()
        print("Resetting")

env.close()
