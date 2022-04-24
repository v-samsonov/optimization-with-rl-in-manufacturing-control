import collections
import numpy as np

from RL_Code.modules.env.simpy_jsp_env.observations_managers.observations_manager_interface import ObservationManagerInterface


class ObservationManagerOrderCentricSort(ObservationManagerInterface):
    def __init__(self, env_obs_dict, norm_obs):
        self.obs_dict, self.obs_true_keys, self.build_obs_dict, self.norm_coefficients = {}, {}, {}, {}
        self.jsp_n_jobs, self.jsp_n_machines = 0, 0
        self.norm_obs = norm_obs
        self.wip_orders_bool_arr, self.wip_machines_bool_arr = np.array([]), np.array([])
        self.last_sim_clock_of_order_change, self.last_sim_clock_of_machine_change = 0, 0

        self.rng = np.random.default_rng(73824)

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
                }
            }

        self.machine_order_nmbr = [int(machine.number) for machine in machine_store_items]

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
                }
            }

        # sort observations
        # order part
        self.build_obs_dict_sorted = {'orders': {}, 'machines': {}, 'rest': self.build_obs_dict['rest']}
        self.indices_oders = np.argsort(self.build_obs_dict['orders']['remaining_time_per_order'])
        # machine part
        self.indices_machines = np.argsort(self.build_obs_dict['machines']['machine_numbers'])
        # sort accordingly
        for key, values in self.build_obs_dict['orders'].items():
            self.build_obs_dict_sorted['orders'][key] = values[self.indices_oders]
        for key, values in self.build_obs_dict['machines'].items():
            self.build_obs_dict_sorted['machines'][key] = values[self.indices_machines]
        a=1


    def build_observations(self):
        # # transofrm observations from remaining times per operation and processing machines into remaining times ordered over machines
        # durations = self.build_obs_dict['orders']['remaining_time_per_operation']
        # machines = self.build_obs_dict['orders']['processing_machine']
        # remaining_processing_time_per_machine = np.array(
        #     list(map(dict(zip(machines, durations)).get, self.machine_order_nmbr)) + [0] * self.padding_size_machines)
        # remaining_processing_time_per_machine[remaining_processing_time_per_machine == None] = 0
        # self.build_obs_dict['machines']['remaining_processing_time_per_machine'] = remaining_processing_time_per_machine
        # sort observations
        self.build_obs_dict_sorted = {'orders': {}, 'machines': {}, 'rest': self.build_obs_dict['rest']}
        for key, values in self.build_obs_dict['orders'].items():
            self.build_obs_dict_sorted['orders'][key] = values[self.indices_oders]
        for key, values in self.build_obs_dict['machines'].items():
            self.build_obs_dict_sorted['machines'][key] = values[self.indices_machines]
        # filter observations
        build_obs_dict_filt = {}
        for key1, val1 in self.obs_true_keys.items():
            build_obs_dict_filt[key1] = {}
            for key2 in val1:
                build_obs_dict_filt[key1][key2] = self.build_obs_dict_sorted[key1][key2]
        # normalize observations
        if self.norm_obs:
            build_obs_dict_filt = self._normalize_observations(obs_dict=build_obs_dict_filt)
        # flatten dict and extract values
        state_all_lst = self._hierarchical_dict_to_list(build_obs_dict_filt)
        state_all_arr = np.concatenate(state_all_lst, axis=0)
        state_all_arr = np.where(state_all_arr < 0, -1, state_all_arr)  # dont normalize negative values and keep them
        # as intended -1
        return state_all_arr.astype(np.float32)  # .astype(int) .astype(np.float32)

    def state_update_on_operation_start(self, order, machine, sim_clock, episode_step):
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

    def state_update_on_operation_end(self, order, sim_clock, episode_step):
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

    def state_update_on_machine_start(self, machine, order, sim_clock, episode_step):
        time_step_machines = sim_clock - self.last_sim_clock_of_machine_change
        self.build_obs_dict['rest']['step_count'] = np.array([episode_step])
        # update parts of states changing only at moments of simulation clock time change
        if time_step_machines > 0:
            self.last_sim_clock_of_machine_change = sim_clock
            self.build_obs_dict['rest']['total_time'] = np.array([sim_clock])
            self.build_obs_dict['machines']['remaining_processing_time_per_machine'][self.wip_machines_bool_arr] = \
                (self.build_obs_dict['machines']['remaining_processing_time_per_machine'][self.wip_machines_bool_arr] - time_step_machines).clip(min=0) # ADDED
        # else:
        #     self.build_obs_dict['machines']['backlog_per_machine'][machine.number] = \
        #         self.build_obs_dict['machines']['backlog_per_machine'][machine.number] - self.last_sim_clock_of_machine_change
        # mark machine as busy (-1->1)
        self.build_obs_dict['machines']['machine_states'][machine.number] = 1
        self.build_obs_dict['machines']['backlog_per_machine'][machine.number] = \
            (self.build_obs_dict['machines']['backlog_per_machine'][machine.number] - order.op_durations_lst[
                order.pointer]).clip(min=0)  # done for consistency, theoretically should be reduced only after processing advanced
        # update times for other orders
        self.build_obs_dict['machines']['remaining_processing_time_per_machine'][machine.number] = order.op_durations_lst[order.pointer] # ADDED

        self.wip_machines_bool_arr[machine.number] = True

    def state_update_on_machine_release(self, machine, sim_clock, episode_step):
        time_step_machines = sim_clock - self.last_sim_clock_of_machine_change
        self.build_obs_dict['rest']['step_count'] = np.array([episode_step])
        # update parts of states changing only at moments of simulation clock time change
        if time_step_machines > 0:
            self.last_sim_clock_of_machine_change = sim_clock
            self.build_obs_dict['rest']['total_time'] = np.array([sim_clock])
            self.build_obs_dict['machines']['remaining_processing_time_per_machine'][self.wip_machines_bool_arr] = \
                (self.build_obs_dict['machines']['remaining_processing_time_per_machine'][self.wip_machines_bool_arr] - time_step_machines).clip(min=0) # ADDED

        # mark machine as idle (1->-1)
        self.build_obs_dict['machines']['machine_states'][machine.number] = -1
        # self.build_obs_dict['machines']['remaining_processing_time_per_machine'][machine.number] = 0 # ADDED
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
