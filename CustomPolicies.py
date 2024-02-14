from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import create_mlp
import torch.nn.functional as F


class CustomSACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space,
                 lr_schedule, net_arch=None,
                 activation_fn=F.relu,
                 *args, **kwargs):

        super(CustomSACPolicy, self).__init__(
            observation_space, action_space,
            lr_schedule, net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=kwargs['features_extractor_class'],
            features_extractor_kwargs=kwargs['features_extractor_kwargs'],
            *args, **kwargs
        )

    def _build(self, lr_schedule):
        # Build the actor and critic networks
        self.net_arch = self._update_net_arch()
        self.actor, self.actor_target = self._build_actor(self.net_arch, lr_schedule)
        self.critic, self.critic_target = self._build_critic(self.net_arch, lr_schedule)

    def _build_actor(self, net_arch, lr_schedule):
        actor_kwargs = dict(net_arch=net_arch['pi'],
                            activation_fn=self.activation_fn,
                            lr_schedule=lr_schedule)
        actor = self.make_actor(**actor_kwargs)
        actor_target = self.make_actor(**actor_kwargs)
        return actor, actor_target

    def _build_critic(self, net_arch, lr_schedule):
        critic_kwargs = dict(net_arch=net_arch['qf'],
                             activation_fn=self.activation_fn,
                             lr_schedule=lr_schedule)
        critic = self.make_critic(**critic_kwargs)
        critic_target = self.make_critic(**critic_kwargs)
        return critic, critic_target

# Usage example
policy_kwargs = {
    "features_extractor_class": LSTMFeaturesExtractor,
    "features_extractor_kwargs": dict(features_dim=64, lstm_hidden_size=64, sequence_length=5),
}

model = SAC(CustomSACPolicy, "your-env", policy_kwargs=policy_kwargs)
