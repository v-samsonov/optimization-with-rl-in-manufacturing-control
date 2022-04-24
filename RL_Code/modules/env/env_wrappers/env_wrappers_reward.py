import numpy as np
from gym.wrappers import TransformReward


class TransformRewardExp(TransformReward):
    def __init__(self, environment, exp_coef, reward_magnitude, **kwargs):
        super(TransformReward, self).__init__(environment)
        self.exp_coef = exp_coef
        self.reward_magnitude = reward_magnitude
        self.f = _reward_calc_exp

    def reward(self, reward):
        return self.f(reward=reward, done=self.done, opt_time=self.env.jsp_task['optimal_time'],
                      scale=self.reward_magnitude, exp_coef=self.exp_coef)


def _reward_calc_exp(reward, done, opt_time, scale, exp_coef):
    if done:
        return scale * (exp_coef ** opt_time) / (exp_coef ** reward)
    else:
        return 0


class LowerBoundMakespanReward(TransformReward):
    # Implementation of reward design from NIPS 2020
    # "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning" paper
    # http://arxiv.org/pdf/2010.12367v1
    def __init__(self, environment, norm_reward_flag, max_length_rew_norm_arr, **kwargs):
        super(TransformReward, self).__init__(environment)
        self.norm_reward_flag = norm_reward_flag
        self.max_length_rew_norm_arr = max_length_rew_norm_arr
        self.rewards_arr = np.array([])
        self.rewards_sum_arr = np.array([])
        self.rew_sum = 0
        self.f = _reward_calc_lower_bound_makespan

    def reward(self, reward):
        self._lower_bound = (self.obs_manager.build_obs_dict['machines']['remaining_processing_time_per_machine'] +
                             self.obs_manager.build_obs_dict['machines']['backlog_per_machine']).max().copy()
        self.time_now = self.obs_manager.build_obs_dict['rest']['total_time'][0].copy()
        if self.ep_step == 1:
            self._lower_bound_init = self._lower_bound.copy()
            self._lower_bound_prev_step = self._lower_bound
            self.time_prev_step = self.obs_manager.build_obs_dict['rest']['total_time'][0].copy()
            # print("init lb:", self._lower_bound_init)

        time_dif = self.time_now - self.time_prev_step
        reward = self.f(reward=reward,
                        _lower_bound_prew_step=self._lower_bound_prev_step,
                        _lower_bound=self._lower_bound, time_dif=time_dif)

        self.time_prev_step = self.time_now.copy()
        self._lower_bound_prev_step = self._lower_bound.copy()
        # if reward != 0:
        self.rewards_arr = np.append(self.rewards_arr, reward)
        if len(self.rewards_arr) > self.max_length_rew_norm_arr:
            self.rewards_arr = np.delete(self.rewards_arr, -1)

        if (self.norm_reward_flag):
            # if reward!=0:
            reward = (reward - self.rewards_arr.mean()) / (self.rewards_arr.std() + 1e-5)

        return reward


def _reward_calc_lower_bound_makespan(reward, _lower_bound_prew_step,
                                      _lower_bound, time_dif):
    lower_bound_makespan = _lower_bound_prew_step - time_dif - _lower_bound  # chosen_operation_time
    if lower_bound_makespan < 0:
        return lower_bound_makespan
    else:
        return 0

# ToDo: https://stable-baselines.readthedocs.io/en/v2.6.0/_modules/stable_baselines/common/vec_env/vec_normalize.html
