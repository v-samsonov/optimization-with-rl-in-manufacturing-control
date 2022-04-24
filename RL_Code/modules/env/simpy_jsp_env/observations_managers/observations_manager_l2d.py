import collections
from builtins import int

import numpy as np

from RL_Code.modules.env.simpy_jsp_env.observations_managers.observations_manager_interface import ObservationManagerInterface


class ObservationManagerL2D(ObservationManagerInterface):
    def __init__(self, env_obs_dict, norm_obs):
        self.obs_dict, self.obs_true_keys, self.build_obs_dict, self.norm_coefficients = {}, {}, {}, {}
        self.jsp_n_jobs, self.jsp_n_machines = 0, 0
        self.norm_obs = norm_obs
        self.wip_orders_bool_arr, self.wip_machines_bool_arr = np.array([]), np.array([])
        self.last_sim_clock_of_order_change, self.last_sim_clock_of_machine_change = 0, 0

        self.posRewards = 0 #dummyvar
        self.initQuality = 0


        # Define what to include into state information
        for k1 in env_obs_dict.keys():
            self.obs_true_keys[k1] = [k2 for k2 in env_obs_dict[k1].keys() if env_obs_dict[k1][k2] is True]

    def reset_state(self, jsp_task, max_n_orders, max_n_machines,
                    order_store_items, machine_store_items):
        self.jsp_n_jobs = jsp_task['n_jobs']
        self.jsp_n_machines = jsp_task['n_resources']
        self.n_ops_per_job = jsp_task['n_ops_per_job']
        self.max_op_time = jsp_task['max_op_time']
        self.max_episode_steps = self.n_ops_per_job * self.jsp_n_jobs
        self.max_n_orders, self.max_n_machines = max_n_orders, max_n_machines
        self.wip_orders_bool_arr = np.repeat(False, self.max_n_orders)
        self.wip_machines_bool_arr = np.repeat(False, self.max_n_machines)
        self.last_sim_clock_of_order_change, self.last_sim_clock_of_machine_change = 0, 0


        self.first_col = np.array([i * self.n_ops_per_job for i in range(self.jsp_n_jobs)])
        candidate = np.array([i * self.n_ops_per_job for i in range(self.jsp_n_jobs)])
        mask = np.full(shape=self.jsp_n_jobs, fill_value=0, dtype=bool)

        self.dur = jsp_task["jssp_instance"]["opduration"].values.reshape(self.jsp_n_jobs, self.jsp_n_machines)
        self.temp1 = np.zeros_like(self.dur, dtype=np.single)
        machine_counter = np.full(shape=self.jsp_n_machines, fill_value=0, dtype=np.int32)

        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max()
        self.max_endTime = self.initQuality

        msk = np.array([0 for _ in range(self.max_episode_steps)])
        fea = np.concatenate((self.LBs.reshape(-1, 1), msk.reshape(-1, 1)), axis=1)

        opIDsOnMchs = -self.jsp_n_jobs * np.ones(shape=(self.jsp_n_machines, self.max_n_orders), dtype=np.int32)

        conj_nei_up_stream = np.eye(self.max_episode_steps, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.max_episode_steps, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[candidate] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[np.array([(i+1) * self.n_ops_per_job - 1 for i in range(self.jsp_n_jobs)])] = 0
        self_as_nei = np.eye(self.max_episode_steps, dtype=np.single)
        adj = self_as_nei + conj_nei_up_stream

        self.norm_coef_cum_time = 1000  #divide by et_normalize_coef

        self.norm_coef_total_max_lb = self.jsp_n_jobs * self.n_ops_per_job * self.max_op_time  # ToDo: in case of
        # multiple job sizes takes the max LB estimation
        self.norm_coef_order_max_backlog = self.n_ops_per_job * self.max_op_time  # ToDo: in case of
        # multiple job sizes takes the max backlog estimation
        self.norm_coef_machine_max_backlog = self.jsp_n_jobs * self.max_op_time  # ToDo: in case of
        # multiple job sizes takes the max backlog estimation
        padding_size_orders = self.max_n_orders - self.jsp_n_jobs
        self.padding_size_machines = self.max_n_machines - self.jsp_n_machines
        self.machine_store_items=machine_store_items
        self.build_obs_dict = \
            {  # ToDo: implement a look ahead for n operations per order as a part of the state space
                'orders': {
                    'order_number': np.array(
                        ([int(order.number) for order in order_store_items] + [-1] * padding_size_orders)),
                    'processing_machine': np.array(
                        [order.processing_machine for order in order_store_items] + [-1] * padding_size_orders),
                    'remaining_time_per_operation': np.array([0] * len(order_store_items) + [0] * padding_size_orders),
                    'remaining_time_per_order': np.array([order.tot_remaining_time for order in order_store_items] +
                                                         [0 for _ in range(padding_size_orders)]),
                    # ==>2."remaining_time"]
                    'next_op_duration_per_order': np.array([order.pending_op_duration for order in order_store_items] +
                                                           [0 for _ in range(padding_size_orders)]),
                    # ==>3."next_duration"
                    'next_machine_type_required': np.array(
                        [order.machine_type_required for order in order_store_items] +
                        [-1 for _ in range(padding_size_orders)]),  # ==>4."next_machine"
                    'processing_started_at': np.array([-1] * self.max_n_orders),
                },
                'machines': {
                    'machine_numbers': np.array(
                        [int(machine.number) for machine in machine_store_items] + [-1 for _ in range(
                            self.padding_size_machines)]),  # index on machine id
                    # # ToDo: implement use case with multiple possible operations per machine
                    'op_capabilities_lst': np.array([machine.op_capability for machine in machine_store_items] +
                                                    [-1 for _ in range(self.padding_size_machines)]),  # index on machine id
                    'remaining_processing_time_per_machine': np.array(
                        [0] * len(machine_store_items) + [0] * self.padding_size_machines),
                    'machine_states': np.array([machine.state for machine in machine_store_items] +
                                               [-1 for _ in range(self.padding_size_machines)]),
                    # index on machine id # ==>1."machine_state"
                    'backlog_per_machine':  # ==>6."queue_time"
                        np.array([sum([order.op_durations_lst[np.in1d(order.op_types_lst, m.op_capability)].sum() for
                                       order in order_store_items])
                                  for m in machine_store_items
                                  ] + [0 for _ in range(self.padding_size_machines)]),
                },  # index on machine id
                'rest': {
                    'total_time': np.array([0]),  # ==>5."total_time"
                    'step_count': np.array([0])
                },
                'l2d': {
                    'adj': adj,
                    'fea': fea,
                    'candidate': candidate,
                    'mask': mask,
                    'opIDsOnMchs': opIDsOnMchs,
                    'opIDsOnMchs_rolled': opIDsOnMchs,
                    'machine_counter': machine_counter,
                    'action': 0
                }
            }
        # # transofrm observations from remaining times per operation and processing machines into remaining times ordered over machines
        self.machine_order_nmbr = [int(machine.number) for machine in machine_store_items]
        # durations = self.build_obs_dict['orders']['remaining_time_per_operation']
        # machines = self.build_obs_dict['orders']['processing_machine']
        # remaining_processing_time_per_machine = np.array(
        #     list(map(dict(zip(machines, durations)).get, self.machine_order_nmbr)) + [0] * self.padding_size_machines)
        # remaining_processing_time_per_machine[remaining_processing_time_per_machine == None] = 0
        # self.build_obs_dict['machines']['remaining_processing_time_per_machine'] = remaining_processing_time_per_machine

        self.norm_coefficients = \
            {'orders': {
                'order_number': np.array(self.max_n_orders * [self.max_n_orders - 1]),
                'processing_machine': np.array(self.max_n_orders * [self.max_n_machines - 1]),
                'remaining_time_per_operation': np.array(self.max_n_orders * [self.max_op_time]),
                'remaining_time_per_order': np.array(self.max_n_orders * [self.norm_coef_order_max_backlog]),
                'next_op_duration_per_order': np.array(self.max_n_orders * [self.max_op_time]),  # ==>3."next_duration"
                'next_machine_type_required': np.array(self.max_n_orders * [self.max_n_machines - 1]),
                # ==>4."next_machine"
                'processing_started_at': np.array(self.max_n_orders * [self.norm_coef_order_max_backlog]),
            },
                'machines': {
                    'machine_numbers': np.array(self.max_n_machines * [self.max_n_machines - 1]),
                    # index on machine id
                    # # ToDo: implement use case with multiple possible operations per machine
                    'op_capabilities_lst': np.array(self.max_n_machines * [self.max_n_machines - 1]),
                    'remaining_processing_time_per_machine': np.array(self.max_n_machines * [self.max_op_time]),
                    # index on machine id
                    'machine_states': np.array(self.max_n_machines * [1]),
                    # index on machine id # ==>1."machine_state"
                    'backlog_per_machine':
                        self.norm_coef_machine_max_backlog,
                    # ==>6."queue_time"
                },  # index on machine id
                'rest': {
                    'total_time': np.array([self.norm_coef_total_max_lb]),  # ==>5."total_time"
                    'step_count': np.array([self.max_episode_steps])
                },
                'l2d': {
                    'adj': 1,
                    'fea': (self.norm_coef_cum_time, 1),
                    'candidate': 1,
                    'mask': 1,
                    'opIDsOnMchs': 1,
                    'opIDsOnMchs_rolled': 1,
                    'machine_counter': 1,
                    'action': 1
                }
            }

    def build_observations(self):
        # filter observations
        build_obs_dict_filt = {}
        for key1, val1 in self.obs_true_keys.items():
            build_obs_dict_filt[key1] = {}
            for key2 in val1:
                build_obs_dict_filt[key1][key2] = self.build_obs_dict[key1][key2]
        # normalize observations
        if self.norm_obs:
            build_obs_dict_filt = self._normalize_observations(obs_dict=build_obs_dict_filt)
        # flatten dict and extract values
        # state_all_lst = self._hierarchical_dict_to_list(build_obs_dict_filt)
        # state_all_arr = np.concatenate(state_all_lst, axis=0)
        # state_all_arr = np.where(state_all_arr < 0, -1, state_all_arr)  # dont normalize negative values and keep them
        return build_obs_dict_filt # .astype(int) .astype(np.float32)

    def state_update_on_operation_start(self, order, machine, sim_clock, episode_step):
        if order.number==4:
            ay=1

        time_step_orders = sim_clock - self.last_sim_clock_of_order_change
        self.build_obs_dict['rest']['step_count'] = np.array([episode_step])
        # update parts of states changing only at moments of simulation clock time chang
        if time_step_orders > 0:
            self.last_sim_clock_of_order_change = sim_clock
            self.build_obs_dict['rest']['total_time'] = np.array([sim_clock])
            # update times for other orders
            self.build_obs_dict['orders']['remaining_time_per_operation'][self.wip_orders_bool_arr] = \
                (self.build_obs_dict['orders']['remaining_time_per_operation'][
                     self.wip_orders_bool_arr] - time_step_orders).clip(min=0)
            # self.build_obs_dict['orders']['remaining_time_per_order'][self.wip_orders_bool_arr] = \
            #     (self.build_obs_dict['orders']['remaining_time_per_order'][
            #          self.wip_orders_bool_arr] - time_step_orders).clip(min=0)
        # change state information related to order
        self.build_obs_dict['orders']['processing_machine'][order.number] = machine.number
        self.build_obs_dict['orders']['remaining_time_per_operation'][order.number] = order.op_durations_lst[
            order.pointer]
        self.build_obs_dict['orders']['remaining_time_per_order'][order.number] = \
            (self.build_obs_dict['orders']['remaining_time_per_order'][order.number] - order.op_durations_lst[
                order.pointer]).clip(
                min=0)  # done for consistency, theoretically should be reduced only after processing advanced
        self.build_obs_dict['orders']['next_op_duration_per_order'][order.number] = order.pending_op_duration
        self.build_obs_dict['orders']['next_machine_type_required'][order.number] = order.machine_type_required
        self.build_obs_dict['orders']['processing_started_at'][order.number] = sim_clock
        # mark order as wip
        self.wip_orders_bool_arr[order.number] = True

        self.build_obs_dict['l2d']['mask'][order.number] = True # mark order as unavailable for scheduling before current opperation ends
        # if order.pointer == order.tot_n_operations - 1:
        #     self.build_obs_dict['l2d']['mask'][order.number] = True
        # else:
        #     self.build_obs_dict['l2d']['candidate'][order.number] = self.build_obs_dict['l2d']['candidate'][
        #                                                                 order.number] + 1

        self.temp1[order.number, order.pointer] = sim_clock + self.dur[order.number, order.pointer]
        self.LBs = calEndTimeLB(self.temp1, self.dur)
        self.max_endTime = self.LBs.max()

        self.build_obs_dict['l2d']['machine_counter'][machine.number] += 1
        self.build_obs_dict['l2d']['opIDsOnMchs'][
            order.op_types_lst[order.pointer], self.build_obs_dict['l2d']['machine_counter'][machine.number]-1] =\
            order.number * order.tot_n_operations + order.pointer
        self.build_obs_dict['l2d']['opIDsOnMchs_rolled'] = np.roll(self.build_obs_dict['l2d']['opIDsOnMchs'], -1, axis=0)
        self.build_obs_dict['l2d']['fea'][order.number * order.tot_n_operations + order.pointer][1] = 1
        self.build_obs_dict['l2d']['fea'][:,0] = self.LBs.reshape(-1, 1).transpose()

        # adj matrix
        action = order.number * order.tot_n_operations + order.pointer
        self.build_obs_dict['l2d']['action'] = action
        precd, succd = self.getActionNbghs(action, self.build_obs_dict['l2d']['opIDsOnMchs_rolled'])
        self.build_obs_dict['l2d']['adj'][action] = 0
        self.build_obs_dict['l2d']['adj'][action, action] = 1
        if action not in self.first_col:
            self.build_obs_dict['l2d']['adj'][action, action - 1] = 1
        self.build_obs_dict['l2d']['adj'][action, precd] = 1
        self.build_obs_dict['l2d']['adj'][succd, action] = 1



    def state_update_on_operation_end(self, order, sim_clock, episode_step):
        if order.number==4:
            ay=1
        time_step_orders = sim_clock - self.last_sim_clock_of_order_change
        self.build_obs_dict['rest']['step_count'] = np.array([episode_step])
        # update parts of states changing only at moments of simulation clock time chang
        if time_step_orders > 0:
            self.last_sim_clock_of_order_change = sim_clock
            self.build_obs_dict['rest']['total_time'] = np.array([sim_clock])
            # update times for other orders
            self.build_obs_dict['orders']['remaining_time_per_operation'][self.wip_orders_bool_arr] = \
                (self.build_obs_dict['orders']['remaining_time_per_operation'][
                     self.wip_orders_bool_arr] - time_step_orders).clip(min=0)
            # self.build_obs_dict['orders']['remaining_time_per_order'][self.wip_orders_bool_arr] = \
            #     (self.build_obs_dict['orders']['remaining_time_per_order'][
            #          self.wip_orders_bool_arr] - time_step_orders).clip(min=0)
        # change state information related to order
        self.build_obs_dict['orders']['processing_machine'][order.number] = -1
        # 'remaining_time_per_operation' was updated above together with other orders
        self.build_obs_dict['orders']['next_op_duration_per_order'][order.number] = order.pending_op_duration
        self.build_obs_dict['orders']['next_machine_type_required'][order.number] = order.machine_type_required
        self.build_obs_dict['orders']['processing_started_at'][order.number] = -1
        # mark order as idle
        self.wip_orders_bool_arr[order.number] = False
        if order.pointer < order.tot_n_operations - 1:
            self.build_obs_dict['l2d']['mask'][order.number] = False
            self.build_obs_dict['l2d']['candidate'][order.number] = self.build_obs_dict['l2d']['candidate'][
                                                                        order.number] + 1

    def state_update_on_machine_start(self, machine, order, sim_clock, episode_step):
        time_step_machines = sim_clock - self.last_sim_clock_of_machine_change
        self.build_obs_dict['rest']['step_count'] = np.array([episode_step])
        # update parts of states changing only at moments of simulation clock time change
        if time_step_machines > 0:
            self.last_sim_clock_of_machine_change = sim_clock
            self.build_obs_dict['rest']['total_time'] = np.array([sim_clock])
            # self.build_obs_dict['machines']['backlog_per_machine'][self.wip_machines_bool_arr] = \
            #     (self.build_obs_dict['machines']['backlog_per_machine'][
            #          self.wip_machines_bool_arr] - time_step_machines)  # .clip(min=0)
        # else:
        #     self.build_obs_dict['machines']['backlog_per_machine'][machine.number] = \
        #         self.build_obs_dict['machines']['backlog_per_machine'][machine.number] - self.last_sim_clock_of_machine_change
        # mark machine as busy (-1->1)
        self.build_obs_dict['machines']['machine_states'][machine.number] = 1
        self.build_obs_dict['machines']['backlog_per_machine'][machine.number] = \
            (self.build_obs_dict['machines']['backlog_per_machine'][machine.number] - order.op_durations_lst[
                order.pointer]).clip(
                min=0)  # done for consistency, theoretically should be reduced only after processing advanced

        self.wip_machines_bool_arr[machine.number] = True


    def state_update_on_machine_release(self, machine, sim_clock, episode_step):
        time_step_machines = sim_clock - self.last_sim_clock_of_machine_change
        self.build_obs_dict['rest']['step_count'] = np.array([episode_step])
        # update parts of states changing only at moments of simulation clock time change
        if time_step_machines > 0:
            self.last_sim_clock_of_machine_change = sim_clock
            self.build_obs_dict['rest']['total_time'] = np.array([sim_clock])
            # self.build_obs_dict['machines']['backlog_per_machine'][self.wip_machines_bool_arr] = \
            #     self.build_obs_dict['machines']['backlog_per_machine'][self.wip_machines_bool_arr] - time_step_machines
        # else:
        #     self.build_obs_dict['machines']['backlog_per_machine'][machine.number] = \
        #         self.build_obs_dict['machines']['backlog_per_machine'][machine.number] - self.last_sim_clock_of_machine_change
        # mark machine as idle (1->-1)
        self.build_obs_dict['machines']['machine_states'][machine.number] = -1
        self.wip_machines_bool_arr[machine.number] = False

    def _normalize_observations(self, obs_dict):
        build_obs_dict_filt_norm = {}
        for key1, dict1 in obs_dict.items():
            build_obs_dict_filt_norm[key1] = {}
            for key2 in dict1.keys():
                build_obs_dict_filt_norm[key1][key2] = obs_dict[key1][key2] / self.norm_coefficients[key1][key2]
        return build_obs_dict_filt_norm

    def _hierarchical_dict_to_list(self, d, parent_key=''):
        obs_list = []
        for k, v in d.items():
            new_key = k
            if isinstance(v, collections.MutableMapping):
                obs_list.extend(self._hierarchical_dict_to_list(v, new_key))
            else:
                obs_list.append(v)
        return obs_list

    def set_optimal_time(self, optimal_time):
        # self.optimal_time = optimal_time
        self.norm_coefficients['rest']['total_time'] = optimal_time

    def getActionNbghs(self, action, opIDsOnMchs):
        coordAction = np.where(opIDsOnMchs == action)
        precd = opIDsOnMchs[coordAction[0], coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]].item()
        succdTemp = opIDsOnMchs[coordAction[0], coordAction[1] + 1 if coordAction[1].item() + 1 < opIDsOnMchs.shape[-1] else coordAction[1]].item()
        succd = action if succdTemp < 0 else succdTemp
        # precedX = coordAction[0]
        # precedY = coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]
        # succdX = coordAction[0]
        # succdY = coordAction[1] + 1 if coordAction[1].item()+1 < opIDsOnMchs.shape[-1] else coordAction[1]
        return precd, succd

def calEndTimeLB(temp1, dur_cp):
    x, y = lastNonZero(temp1, 1, invalid_val=-1)
    dur_cp[np.where(temp1 != 0)] = 0
    dur_cp[x, y] = temp1[x, y]
    temp2 = np.cumsum(dur_cp, axis=1)
    temp2[np.where(temp1 != 0)] = 0
    ret = temp1 + temp2
    return ret

def lastNonZero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)
    xAxis = np.arange(arr.shape[0], dtype=np.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet