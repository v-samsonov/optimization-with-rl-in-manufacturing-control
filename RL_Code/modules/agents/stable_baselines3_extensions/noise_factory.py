import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from RL_Code.modules.agents.noise_factory_abs import ActionNoiseConstructorInterface


# get agents's exploration parameters
class ActionNoiseConstructor(ActionNoiseConstructorInterface):
    def build(self, actions_noise_flag, rl_framework_tag, environment, **kwargs):
        builder = self._get_noise_constructor(actions_noise_flag, rl_framework_tag)
        return builder(environment, **kwargs)

    def _get_noise_constructor(self, actions_noise_flag, rl_framework):
        if actions_noise_flag is None:
            return self.no_noise
        if rl_framework == 'stable_baselines3':
            if actions_noise_flag == 'OrnsteinUhlenbeckActionNoise':
                return self.ornstein_uhlenbeck_action_noise
            elif actions_noise_flag == 'NormalActionNoise':
                return self.normal_action_noise
            else:
                raise ValueError(f'build method for {actions_noise_flag} in {rl_framework} is undefined')

        else:
            raise ValueError(f'{rl_framework} has no noise methods defined')

    def ornstein_uhlenbeck_action_noise(self, environment, actions_noise_sigma, **kwargs):
        n_actions = environment.action_space.shape[0]
        return OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                            sigma=float(actions_noise_sigma) * np.ones(n_actions))

    def normal_action_noise(self, environment, actions_noise_sigma, **kwargs):
        n_actions = len(environment.action_space())
        return NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=float(actions_noise_sigma) * np.ones(n_actions))

    def no_noise(self, *args, **kwargs):
        return None
