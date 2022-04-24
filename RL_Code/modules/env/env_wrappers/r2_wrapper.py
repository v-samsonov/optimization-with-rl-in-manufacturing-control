import numpy as np


class RankedRewardsBuffer:  # todo replace lists with numpy
    def __init__(self, size, percentile, too_good_penalty, less_rew_better_flag, negative_rew_flag):
        self.buffer_max_length_per_task = size
        self.percentile = percentile
        self.too_good_penalty = too_good_penalty
        self.less_rew_better_flag = less_rew_better_flag
        self.negative_rew_flag = negative_rew_flag
        self.buffer_dict = {}
        self.buffer_dict_all = {}
        self.best_seen_rewards_dict = {}

    def add_reward(self, reward, task_key):
        if task_key in self.buffer_dict.keys():
            if len(self.buffer_dict[task_key]) < self.buffer_max_length_per_task:
                self.buffer_dict[task_key].append(reward)
            else:
                self.buffer_dict[task_key] = self.buffer_dict[task_key][1:] + [reward]
            self.buffer_dict_all[task_key].append(reward)
        else:
            self.buffer_dict[task_key] = [reward]
            self.buffer_dict_all[task_key] = [reward]
        if self.less_rew_better_flag:
            self.best_seen_rewards_dict.update({task_key: min(self.buffer_dict_all[task_key])})
        else:
            self.best_seen_rewards_dict.update({task_key: max(self.buffer_dict_all[task_key])})

    def calculate_reward(self, reward, task_key, env, no_precalculated_optimum=True,
                         ):
        # make sure reward has a small offset to avoid division by zero
        if self.negative_rew_flag:
            reward -= 0.001
        else:
            reward += 0.001
        self.add_reward(reward, task_key)
        rewards_lst = self.buffer_dict[task_key]
        if no_precalculated_optimum:
            best_seen_reward = self.best_seen_rewards_dict[task_key]
        else:
            best_seen_reward = env.optimal_makespan

        # if (self.less_rew_better_flag is False) & (self.negative_rew_flag is False):  # Exp_Reward
        #     scaled_rewards = [obs_reward / best_seen_reward for obs_reward in rewards_lst]
        # else: # makespan as reward
        #     scaled_rewards = [best_seen_reward / obs_reward for obs_reward in rewards_lst]

        if self.less_rew_better_flag & (self.negative_rew_flag is False):  # makespan as reward
            scaled_rewards = [best_seen_reward / obs_reward for obs_reward in rewards_lst]
        elif (self.less_rew_better_flag is False) & (self.negative_rew_flag is False):  # Exp_Reward
            scaled_rewards = [obs_reward / best_seen_reward for obs_reward in rewards_lst]
        elif (
                self.less_rew_better_flag is False) & self.negative_rew_flag:  # ReversedStepwiseMakespanReward or LowerBoundMakespanReward
            scaled_rewards = [best_seen_reward / obs_reward for obs_reward in rewards_lst]
        else:
            raise ValueError(
                f"combination of less_rew_better_flag {self.less_rew_better_flag} and negative_rew_flag {self.negative_rew_flag} is unexpected")

        scaled_reward = scaled_rewards[-1]
        reward_threshold = np.percentile(scaled_rewards, self.percentile)

        # print('rewards_lst', self.buffer_dict_all[task_key], 'min_seen_reward', min_seen_reward, 'scaled_rewards', scaled_rewards, 'reward_threshold', reward_threshold)

        if scaled_reward < reward_threshold:
            return -1.0
        elif scaled_reward == reward_threshold:
            # print("reward_threshold", reward_threshold)
            # print('too_good_penalty')
            return np.random.choice([-1, 1], p=[self.too_good_penalty, 1 - self.too_good_penalty])
        else:
            return 1.0


def get_r2_env_wrapper(r2_buffer_max_length_per_task, r2_percentile, too_good_penalty, less_rew_better_flag,
                       negative_rew_flag, **kwargs):
    class RankedRewardsEnvWrapper():
        def __init__(self, env):
            # super(JSPEnv, self)
            self.env = env
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            self.reward_range = self.env.reward_range
            self.metadata = self.env.metadata
            self.spec = self.env.spec
            self.r2_buffer = RankedRewardsBuffer(r2_buffer_max_length_per_task, r2_percentile, too_good_penalty,
                                                 less_rew_better_flag, negative_rew_flag)
            # print('ranked reward buffer is initialized')

        def reset_percentile(self, percentile):
            self.r2_buffer.percentile = percentile

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            if done:
                # print('reward', reward)
                reward = self.r2_buffer.calculate_reward(reward, self.env.jsp_id, self.env)
                # print('r2', reward)
            else:
                return obs, 0, done, info
            return obs, reward, done, info

        def reset(self, jsp_ind=None):
            return self.env.reset(jsp_ind=jsp_ind)

        def seed(self, *args):
            self.env.seed(*args)

        def close(self):
            self.env.close()

    return RankedRewardsEnvWrapper


class ExpRankedRewardsBuffer:  # todo replace lists with numpy
    def __init__(self, size, percentile, too_good_penalty, reward_magnitude, exp_coef):
        self.buffer_max_length_per_task = size
        self.percentile = percentile
        self.too_good_penalty = too_good_penalty
        self.buffer_dict = {}
        self.buffer_dict_all = {}
        self.best_seen_rewards_dict = {}
        self.reward_magnitude = reward_magnitude
        self.exp_coef = exp_coef

    def add_reward(self, reward, task_key):
        if task_key in self.buffer_dict.keys():
            if len(self.buffer_dict[task_key]) < self.buffer_max_length_per_task:
                self.buffer_dict[task_key].append(reward)
            else:
                self.buffer_dict[task_key] = self.buffer_dict[task_key][1:] + [reward]
            self.buffer_dict_all[task_key].append(reward)
        else:
            self.buffer_dict[task_key] = [reward]
            self.buffer_dict_all[task_key] = [reward]

        self.best_seen_rewards_dict.update({task_key: max(self.buffer_dict_all[task_key])})

    def calculate_reward(self, reward, task_key, env, no_precalculated_optimum=True,
                         ):
        self.add_reward(reward, task_key)
        rewards_lst = self.buffer_dict[task_key]
        # best_seen_reward = self.best_seen_rewards_dict[task_key]
        reward_threshold = np.percentile(rewards_lst,
                                         100 - self.percentile)  # ToDo: implement less_rew_better_flag case with normalization
        # print("reward_threshold", reward_threshold)
        exp_reward = self.reward_magnitude * (self.exp_coef ** reward_threshold) / (self.exp_coef ** reward)
        # print('rewards_lst', self.buffer_dict_all[task_key], 'min_seen_reward', min_seen_reward, 'scaled_rewards', scaled_rewards, 'reward_threshold', reward_threshold)

        return np.clip(exp_reward, 0, self.reward_magnitude)


def get_exp_r2_env_wrapper(r2_buffer_max_length_per_task, r2_percentile, too_good_penalty,
                           reward_magnitude, exp_coef, **kwargs):
    class RankedRewardsEnvWrapper():
        def __init__(self, env):
            # super(JSPEnv, self)
            self.env = env
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            self.reward_range = self.env.reward_range
            self.metadata = self.env.metadata
            self.spec = self.env.spec
            self.r2_buffer = ExpRankedRewardsBuffer(r2_buffer_max_length_per_task, r2_percentile, too_good_penalty,
                                                    reward_magnitude, exp_coef)
            # print('ranked reward buffer is initialized')

        def reset_percentile(self, percentile):
            self.r2_buffer.percentile = percentile

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            if done:
                # print('reward', reward)
                reward = self.r2_buffer.calculate_reward(reward, self.env.jsp_id, self.env)
                # print('r2', reward)
            else:
                return obs, 0, done, info
            return obs, reward, done, info

        def reset(self, jsp_ind=None):
            return self.env.reset(jsp_ind=jsp_ind)

        def seed(self, *args):
            self.env.seed(*args)

        def close(self):
            self.env.close()

    return RankedRewardsEnvWrapper
