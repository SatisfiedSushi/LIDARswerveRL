import os
import gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomActor(nn.Module):
    def __init__(self, observation_space, action_space, feature_dim):
        super(CustomActor, self).__init__()
        # Customize your actor network here
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.shape[0])
        )

    def forward(self, x):
        return th.tanh(self.net(x))


class CustomCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomCritic, self).__init__()
        # Customize your critic network here
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0] + action_space.shape[0], 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, action):
        return self.net(th.cat([x, action], dim=1))


class CustomDDPG(DDPG):
    def _setup_model(self):
        super(CustomDDPG, self)._setup_model()
        self.actor = CustomActor(self.observation_space, self.action_space,
                                 self.policy_kwargs.get("features_extractor_class", None))
        self.actor_target = CustomActor(self.observation_space, self.action_space,
                                        self.policy_kwargs.get("features_extractor_class", None))
        self.critic = CustomCritic(self.observation_space, self.action_space)
        self.critic_target = CustomCritic(self.observation_space, self.action_space)

        # Copy the weights from the actor/critic to their targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


# Initialize environment
environment = gym.make('Pendulum-v0')

# Define action noise for DDPG
n_actions = environment.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize custom DDPG model
model = CustomDDPG("MlpPolicy", environment, action_noise=action_noise, verbose=1)

# Now you can train, evaluate and use the model as usual
