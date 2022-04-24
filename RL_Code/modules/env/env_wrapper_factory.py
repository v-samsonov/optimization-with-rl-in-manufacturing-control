# from gym import wrappers
# import wandb

from RL_Code.modules.env.env_wrappers.env_wrappers_action import TransformActionOpRelDuration, L2DtoSimpyActionWrapper, \
    TransformActionOrder
from RL_Code.modules.env.env_wrappers.env_wrappers_obs import FlattenObservation, L2DObservation
from RL_Code.modules.env.env_wrappers.env_wrappers_reward import TransformRewardExp, \
    LowerBoundMakespanReward
from RL_Code.modules.env.env_wrappers.r2_wrapper import get_r2_env_wrapper, get_exp_r2_env_wrapper
from RL_Code.modules.gym_modifications.monitor import Monitor_Mod


class EnvWrapperConstructor():
    def __init__(self):
        self.less_rew_better_flag = True  # corresponds to makespan as reward
        self.negative_rew_flag = False  # corresponds to makespan as reward

    # Client component
    def build(self, environment, wrapper_tag, *args, **kwargs):
        wrapped_env = self._get_builder(wrapper_tag)
        return wrapped_env(environment, **kwargs)

    # Creator component
    def _get_builder(self, wrapper_tag):
        if wrapper_tag == 'Gym_Monitor':
            return self._wrap_in_gym_monitor
        elif wrapper_tag == 'Exp_Reward':
            self.less_rew_better_flag = False
            self.negative_rew_flag = False
            return self._wrap_in_transform_reward_exp
        elif wrapper_tag == 'OperationRelDuration':
            return self._wrap_in_transform_action_op_rel_duration_discrete
        elif wrapper_tag == 'TransformActionOrder':
            return self._wrap_in_transform_action_order
        elif wrapper_tag == 'Ranked_Reward':
            return self._wrap_in_ranked_reward
        elif wrapper_tag == 'Exp_Ranked_Reward':
            return self._wrap_in_exp_ranked_reward
        elif wrapper_tag == 'LowerBoundMakespanReward':
            self.less_rew_better_flag = False
            self.negative_rew_flag = True
            return self._wrap_in_lower_bound_makespanReward
        elif wrapper_tag == 'FlattenObservation':
            return self._wrap_in_flatten_observation
        elif wrapper_tag == 'L2DObservation':
            return self._wrap_in_l2d_observation
        elif wrapper_tag == 'L2DtoSimpyActionWrapper':
            return self._wrap_in_l2d_to_simpy_action
        else:
            raise ValueError('wrapper', wrapper_tag, 'is not implemented')

    # Product component
    def _wrap_in_gym_monitor(self, environment, log_path, log_name_suffix, wandb_logs, **kwargs):
        """
        logs episode statistics from an openai gym
        :param environment: openai gym
        :param log_path: path to the logging folder
        :param kwargs:
        :return monitor wrapper with openai gym environment
        """
        # wandb.init(monitor_gym=True)
        environment = Monitor_Mod(environment, log_path, uid=log_name_suffix, wandb_logs=wandb_logs,
                                  resume=True)  # force=True)
        return environment

    def _wrap_in_transform_reward_exp(self, environment, exp_coef, reward_magnitude, *args, **kwargs):
        """
        :param environment: openai gym
        :param exp_coef: steepness of the reward function
        :param reward_magnitude: max reward value
        :param kwags:
        :return: openai gym environment with exponential sparse reward
        """
        environment = TransformRewardExp(environment, exp_coef, reward_magnitude)
        return environment

    def _wrap_in_transform_action_op_rel_duration_discrete(self, environment, *args, **kwargs):
        """
        :param environment: openai gym
        :param kwags:
        :return: openai gym environment with transformed action space
        """
        environment = TransformActionOpRelDuration(environment, **kwargs)
        return environment

    def _wrap_in_transform_action_order(self, environment, *args, **kwargs):
        """
        :param environment: openai gym
        :param kwags:
        :return: openai gym environment with transformed action space
        """
        environment = TransformActionOrder(environment, **kwargs)
        return environment

    def _wrap_in_ranked_reward(self, environment, r2_buffer_max_length_per_task, r2_percentile, too_good_penalty,
                               *args, **kwargs):
        """
        :param environment: openai gym
        :param r2_buffer_max_length_per_task: on how many solutions per instance should current agents performance be evaluated
        :param r2_percentile: threshold to give or not the reward
        :param kwargs:
        :return: environment with r2 reward
        """
        r2_wrapper = get_r2_env_wrapper(r2_buffer_max_length_per_task, r2_percentile, too_good_penalty,
                                        self.less_rew_better_flag, self.negative_rew_flag)
        environment = r2_wrapper(environment)
        return environment

    def _wrap_in_exp_ranked_reward(self, environment, r2_buffer_max_length_per_task, r2_percentile, too_good_penalty,
                                   reward_magnitude, exp_coef, *args, **kwargs):
        exp_r2_wrapper = get_exp_r2_env_wrapper(r2_buffer_max_length_per_task, r2_percentile, too_good_penalty,
                                                reward_magnitude, exp_coef)
        environment = exp_r2_wrapper(environment)
        return environment

    def _wrap_in_lower_bound_makespanReward(self, environment, *args, **kwargs):
        environment = LowerBoundMakespanReward(environment, **kwargs)
        return environment

    def _wrap_in_flatten_observation(self, environment, *args, **kwargs):
        environment = FlattenObservation(environment)
        return environment

    def _wrap_in_l2d_observation(self, environment, *args, **kwargs):
        environment = L2DObservation(environment)
        return environment

    def _wrap_in_l2d_to_simpy_action(self, environment, *args, **kwargs):
        environment = L2DtoSimpyActionWrapper(environment, **kwargs)
        return environment