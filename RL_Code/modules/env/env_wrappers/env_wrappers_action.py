import gym
import numpy as np
import pandas as pd
from gym import spaces

pd.set_option('display.max_columns', 50)


class TransformActionOpRelDuration(gym.ActionWrapper):
    def __init__(self, env, discrete_action_space=True, n_durations_intervals=None, **kwargs):
        super(TransformActionOpRelDuration, self).__init__(env)
        self.rel_action_duration_flag = True
        self.n_durations_intervals = n_durations_intervals
        env.get_machine_by_op_type = True
        min_op_duration_lst, max_op_duration_lst = [], []
        for jsp_task in env.jsp_lst:
            max_op_duration_lst.append(jsp_task["max_op_time"])
            try:  # check if jsp instance has a specified duration of min operation time, else its assumed to be 1
                min_op_duration_lst.append(jsp_task["min_op_time"])
            except:
                min_op_duration_lst.append(1)
        self.min_op_duration = min(min_op_duration_lst)
        self.max_op_duration = max(max_op_duration_lst)

        if n_durations_intervals == None:
            self.op_duration_range = self.max_op_duration - self.min_op_duration
        else:
            self.op_duration_range = n_durations_intervals

        if discrete_action_space:
            self.action_space = spaces.Discrete(self.op_duration_range + 1)
        else:
            self.action_space = spaces.Box(low=np.array([self.min_op_duration]),
                                           high=np.array([self.max_op_duration]),
                                           dtype=np.float32
                                           )

    def action(self, action):
        # calculate order-machine pair based on the relative time
        assert self.action_space.contains(action)
        # get free machines
        free_orders_mask_arr = ~self.obs_manager.wip_orders_bool_arr
        next_machines_4_free_orders = self.obs_manager.build_obs_dict['orders']['next_machine_type_required'][
            free_orders_mask_arr]

        free_machines_mask_arr = ~self.obs_manager.wip_machines_bool_arr
        free_machines_nmbrs_arr = self.obs_manager.build_obs_dict['machines']['machine_numbers'][free_machines_mask_arr]
        free_required_machines_mask_arr = np.in1d(next_machines_4_free_orders, free_machines_nmbrs_arr)
        free_required_machines_nmbrs_arr = next_machines_4_free_orders[free_required_machines_mask_arr]
        free_required_machines_nmbrs_arr = free_required_machines_nmbrs_arr[
            free_required_machines_nmbrs_arr != -1]  # make sure dummy machines used for state space augmentation are not considered

        # pick first machine to start planning with a the given simulation timestep (sequence does not matter)
        chosen_machine_nmbr = np.unique(free_required_machines_nmbrs_arr)[0]
        op_durations_4_machine_mask_arr = (
                self.obs_manager.build_obs_dict['orders']['next_machine_type_required'] == chosen_machine_nmbr)
        available_op_durations_4_machine_mask_arr = np.logical_and(op_durations_4_machine_mask_arr,
                                                                   free_orders_mask_arr)
        available_op_durations_4_machine_arr = self.obs_manager.build_obs_dict['orders']['next_op_duration_per_order'][
            available_op_durations_4_machine_mask_arr]

        # calculate which ops refer to rel_time
        rel_time = action / self.op_duration_range
        preferred_op_dur = available_op_durations_4_machine_arr.min() + rel_time * (
                available_op_durations_4_machine_arr.max() - available_op_durations_4_machine_arr.min())
        # get operation with closes available time
        op_time_deviation_arr = np.absolute(
            self.obs_manager.build_obs_dict['orders']['next_op_duration_per_order'] - preferred_op_dur)
        op_deviation_val = op_time_deviation_arr[available_op_durations_4_machine_mask_arr].min()
        chosen_op_mask_arr = op_time_deviation_arr == op_deviation_val
        final_mask_arr = np.logical_and(available_op_durations_4_machine_mask_arr, chosen_op_mask_arr)
        if np.sum(final_mask_arr) == 1:
            chosen_order_nmbr = self.obs_manager.build_obs_dict['orders']['order_number'][final_mask_arr][0]
        else:  # more then one operations can be chosen
            remaining_order_dur_arr = self.obs_manager.build_obs_dict['orders']['remaining_time_per_order'] - \
                                      self.obs_manager.build_obs_dict['orders']['next_op_duration_per_order']
            max_remaining_dur_avail_order = remaining_order_dur_arr[final_mask_arr].max()
            avail_order_with_max_duration_mask_arr = remaining_order_dur_arr == max_remaining_dur_avail_order
            final_mask_arr = np.logical_and(avail_order_with_max_duration_mask_arr, final_mask_arr)
            if np.sum(final_mask_arr) == 1:
                chosen_order_nmbr = self.obs_manager.build_obs_dict['orders']['order_number'][final_mask_arr][0]
            else:  # more then one operations can be chosen
                min_spt_value = self.obs_manager.build_obs_dict['orders']['next_op_duration_per_order'][
                    final_mask_arr].min()
                min_spt_value_mask_arr = self.obs_manager.build_obs_dict['orders'][
                                             'next_op_duration_per_order'] == min_spt_value
                final_mask_arr = np.logical_and(min_spt_value_mask_arr, final_mask_arr)
                chosen_order_nmbr = self.obs_manager.build_obs_dict['orders']['order_number'][final_mask_arr][0]

        return [chosen_order_nmbr, chosen_machine_nmbr]


class TransformActionOrder(gym.ActionWrapper):
    # action space directly mapped to the order index in the sate space represenattion:
    # self.build_obs_dict['orders']['order_number']
    def __init__(self, env, **kwargs):
        super(TransformActionOrder, self).__init__(env)
        n_actions = self.env.obs_manager.max_n_orders
        self.action_space = spaces.Discrete(n_actions)

    def action(self, action):
        # calculate order-machine pair based on the relative time
        assert self.action_space.contains(action)
        chosen_order_nmbr = action
        chosen_machine_nmbr = self.obs_manager.build_obs_dict['orders']['next_machine_type_required'][action]

        return [chosen_order_nmbr, chosen_machine_nmbr]


class L2DtoSimpyActionWrapper(gym.ActionWrapper):
    # action[0] - order_number, action[1] - op_capability
    # map action space in shape [0, ... n_machines*n_jobs] to [n_jobs, n_machines]
    def __init__(self, env, max_n_orders, max_n_machines, discrete_action_space, *args, **kwargs):
        super(L2DtoSimpyActionWrapper, self).__init__(env)
        assert discrete_action_space  # orders and machine mappings works only for discrete action spaces
        self.action_mapper_dict = {}
        counter = 0
        for job in range(max_n_orders):
            for operation in range(max_n_machines):  # assuming n_operations=n_machines
                self.action_mapper_dict[counter] = [job, operation]
                counter += 1
        self.action_space = spaces.Discrete(len(self.action_mapper_dict))

    def action(self, action):
        action_order = self.action_mapper_dict[action][0]
        action_machine = self.env.obs_manager.build_obs_dict["orders"]["next_machine_type_required"][action_order]
        return [action_order, action_machine]


class RescaleAction(gym.ActionWrapper):
    r"""Rescales the continuous action space of the environment to a range [a,b].
    Example::
        >> RescaleAction(env, a, b).action_space == Box(a,b)
        True
    """

    def __init__(self, env, a, b):
        assert isinstance(env.action_space, spaces.Box), (
            "expected Box action space, got {}".format(type(env.action_space)))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
        self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
        self.action_space = spaces.Box(low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype)

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.a) / (self.b - self.a))
        action = np.clip(action, low, high)
        return action
