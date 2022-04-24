import copy
from collections import namedtuple
from operator import attrgetter

import gym
import numpy as np
import simpy
from gym import spaces
from gym.utils import seeding
from recordtype import recordtype

from RL_Code.modules.env.simpy_jsp_env.observations_managers.observations_manager_l2d import ObservationManagerL2D
from RL_Code.modules.env.simpy_jsp_env.observations_managers.observations_manager_l2d_pick_order import \
    ObservationManagerL2DPickOrder
from RL_Code.modules.env.simpy_jsp_env.observations_managers.observations_manager_order_centric import \
    ObservationManagerOrderCentric
from RL_Code.modules.env.simpy_jsp_env.observations_managers.observations_manager_order_centric_shuffle import \
    ObservationManagerOrderCentricShuffle
from RL_Code.modules.env.simpy_jsp_env.observations_managers.observations_manager_order_centric_shuffle_every_step import \
    ObservationManagerOrderCentricShuffleEveryStep
from RL_Code.modules.env.simpy_jsp_env.observations_managers.observations_manager_order_centric_sort import \
    ObservationManagerOrderCentricSort


class JobShop(gym.Env):
    # noinspection PyTypeChecker
    def __init__(self, jsp_lst, max_n_machines, max_n_orders, env_obs_dict, seed=0, repetitions_per_jsp_task=1,
                 random_jsp_iter=True, norm_obs=True, obs_manager_tag='order_centric', action_mask=False, **kwargs):

        # env metadata
        sim_spec_id_placeholder = namedtuple('id_var', 'id')
        self.spec = sim_spec_id_placeholder('job_shop_env_v1')
        self._type = 't'
        # Fix random seeds
        self.seed(seed=seed)
        # initialize variables
        self.finished_n_orders, self.episode_counter, self.ep_step, self.reset_counter, self.jsp_ind = 0, 0, 0, 0, 0
        self.random_jsp_iter = random_jsp_iter
        self.repetitions_per_jsp_task = repetitions_per_jsp_task
        self.last_sim_clock_of_order_change, self.last_sim_clock_of_machine_change = 0, 0
        # self.max_episode_steps = max_episode_steps
        self.jsp_lst = jsp_lst
        self.get_machine_by_op_type = False  # flag if to search a machine form machine store by number of op
        # capability. Controlled by the TransformActionOpRelDurationDiscrete wrapper
        self.reset_jsp(jsp_ind=None)
        self.init_simpy_env(self.jsp_instance, max_n_machines, max_n_orders)

        if obs_manager_tag == 'l2d':
            self.obs_manager = ObservationManagerL2D(env_obs_dict=env_obs_dict, norm_obs=norm_obs)
        if obs_manager_tag == 'l2d_pick_order':
            self.obs_manager = ObservationManagerL2DPickOrder(env_obs_dict=env_obs_dict, norm_obs=norm_obs)
        elif obs_manager_tag == 'order_centric':
            self.obs_manager = ObservationManagerOrderCentric(env_obs_dict=env_obs_dict, norm_obs=norm_obs,
                                                              action_mask=action_mask)
        elif obs_manager_tag == 'order_centric_shuffled':
            self.obs_manager = ObservationManagerOrderCentricShuffle(env_obs_dict=env_obs_dict, norm_obs=norm_obs,
                                                                     action_mask=action_mask)
        elif obs_manager_tag == 'order_centric_shuffled_every_step':
            self.obs_manager = ObservationManagerOrderCentricShuffleEveryStep(env_obs_dict=env_obs_dict,
                                                                              norm_obs=norm_obs)
        elif obs_manager_tag == 'order_centric_sorted':
            self.obs_manager = ObservationManagerOrderCentricSort(env_obs_dict=env_obs_dict, norm_obs=norm_obs)
        else:
            raise ValueError(f'observation manager {obs_manager_tag} is not implemented')

        self.obs_manager.reset_state(jsp_task=self.jsp_task, max_n_orders=self.max_n_orders,
                                     max_n_machines=self.max_n_machines,
                                     order_store_items=self.order_store.items,
                                     machine_store_items=self.machine_store.items)
        state = self.obs_manager.build_observations()

        self.action_space = spaces.Box(low=np.array([0, 0]),
                                       high=np.array([max_n_orders, max_n_machines]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.repeat(-np.inf, len(state)),
                                            high=np.repeat(np.inf, len(state)),
                                            dtype=np.float32)

    def reset(self, jsp_ind=None):
        # print('RESET ENV')
        # pick a random jsp problem instance from the jsp list for training
        self.reset_jsp(jsp_ind=jsp_ind)
        self.reset_counter += 1
        self.episode_counter += 1
        self.finished_n_orders, self.ep_step = 0, 0
        self.init_simpy_env(jsp_instance=self.jsp_instance, max_n_machines=self.max_n_machines,
                            max_n_orders=self.max_n_orders)
        # reset parts of state information
        _ = self.obs_manager.reset_state(jsp_task=self.jsp_task, max_n_orders=self.max_n_orders,
                                         max_n_machines=self.max_n_machines,
                                         order_store_items=self.order_store.items,
                                         machine_store_items=self.machine_store.items)
        self.obs_manager.set_optimal_time(optimal_time=self.optimal_time)
        state = self.obs_manager.build_observations()
        return state

    def done(self):
        return self.done

    def step(self, action):
        order_number, op_capability = action[0], action[1]
        # print('step', action, 'time', self.env.now)
        # print('mask', self.obs_manager.build_obs_dict['l2d']['mask'])
        self.done = False
        self.ep_step += 1
        # ToDo: add into state space queued but not started orders (not available and stored in buffer)
        self.env.process(self.processing(order_number, op_capability))  # start processing of an order
        # print('queue: ', self.env._queue)
        # run simulation to the next state requiring an action:
        while True:
            self.env.run(until=self.action_opportunity)
            open_tasks_flag = self.check_for_open_tasks()
            if open_tasks_flag is True:  # an action from agent is needed
                # print('asking for action at', self.env.now)
                break

            if (self.finished_n_orders == self.jsp_n_jobs) or (self.ep_step == self.max_episode_steps):
                # print('terminal state')
                self.env.run()  # make sure no events are left unprocessed
                self.done = True
                # print('sim run done at', self.env.now)
                # print('jsp isntance: ', self.jsp_task['jssp_identification'])
                break

        # build state observations: elements in order and machine stores, orders in que, orders in process
        state = self.obs_manager.build_observations()

        # calculate reward
        reward = self.env.now  # reward calculation logic is implemented in an env wrapper
        jsp_ident_str = self.jsp_task['jssp_identification']
        jsp_index = jsp_ident_str[jsp_ident_str.find('_') + 1:jsp_ident_str.rfind('_')]
        info_dict = {'jsp_serial_number': int(jsp_index),
                     'makespan': self.env.now,
                     'optimal_time': self.optimal_time,
                     'action': action, }

        if ('mask' in self.obs_manager.build_obs_dict) & (self.obs_manager.build_obs_dict['mask'] is not None):
            info_dict['action_mask'] = list(np.invert(self.obs_manager.build_obs_dict['mask']))  # .astype('int')
        else:
            info_dict['action_mask'] = None

        return state, reward, self.done, info_dict

    # ---------> Help functions <---------

    def processing(self, order_number, machine_ident):
        # print(self.observations_df_orders, '\n', self.observations_df_machines)
        # get machine and order
        # ToDo: od observation containing que of requested orders amd machines hta tare not available now
        if self.get_machine_by_op_type:
            machine = yield self.machine_store.get(lambda machine: machine.op_capability == machine_ident)
        else:
            # print("\t seizing machine", machine_ident, self.env.now)
            machine = yield self.machine_store.get(lambda machine: machine.number == machine_ident)
            # print("\t seized machine", machine_ident, self.env.now)
        # machine.state = 1
        # print("\t seizing order", order_number, self.env.now)
        order = yield self.order_store.get(lambda order: order.number == order_number)
        # print("\t seized order", order_number, self.env.now)

        # if order_number==2 und machine_ident==2
        # update order and machine data at the start of the processing
        if order.op_types_lst[order.pointer] == machine.op_capability:
            # print('got order', order.number, 'at', self.env.now)
            operation_time = order.op_durations_lst[order.pointer]
            if order.pointer == order.tot_n_operations - 1:  # last operation of the order is finished
                last_operation_flag = True
                order.pending_op_duration = 0
                order.machine_type_required = -1
            else:  # there are still pending operations
                last_operation_flag = False
                order.pending_op_duration = order.op_durations_lst[order.pointer + 1]
                order.machine_type_required = order.op_types_lst[order.pointer + 1]
        else:
            raise ValueError(f'\t order {order_number} and machine {machine_ident} do not match')
        self.obs_manager.state_update_on_machine_start(machine=machine, order=order, sim_clock=self.env.now,
                                                       episode_step=self.ep_step)
        self.obs_manager.state_update_on_operation_start(order=order, machine=machine, sim_clock=self.env.now,
                                                         episode_step=self.ep_step)  # important to have updated machine
        # state before the order state update for obs_manger logic

        # print(order)
        # print('order', order_number, 'machine', machine_ident)
        # print('mask', self.obs_manager.build_obs_dict['l2d']['mask'])
        # #print('candidate', self.obs_manager.build_obs_dict['l2d']['candidate'])
        # if order.op_types_lst[order.pointer] != machine.op_capability:
        #     print('machine_needed', order.op_types_lst[order.pointer], 'machine_requested', machine.op_capability)
        #     print(self.obs_manager.build_obs_dict["orders"])

        # # ask agent if more orders should be scheduled before processing starts
        # if self.check_for_open_tasks(): # othervise simulation will run without schedulingthe last available order
        self.action_opportunity.succeed()
        self.action_opportunity = self.env.event()

        # process fetched order on chosen machine
        self.chosen_operation_time = operation_time  # communicates durations to ReversedStepwiseMakespanReward Wrapper
        processing_event = self.env.timeout(operation_time, value=machine.number)
        yield processing_event

        if last_operation_flag:
            order.op_durations_lst[order.pointer] = 0  # important to set operation duration to 0 after processing,
            # otherwise the remaining time will be estimated wrong
            order.done = True
            order.tot_remaining_time = 0
            self.finished_n_orders += 1
        else:
            order.op_durations_lst[order.pointer] = 0
            order.pointer += 1

        # release resources back to stores
        # print('returned machine', machine.number, 'at', self.env.now)
        # print('returned order', order.number, 'at', self.env.now)
        # machine.state = - 1
        yield self.machine_store.put(machine)
        # print("\t releasing machine", machine_ident, self.env.now)
        self.obs_manager.state_update_on_machine_release(machine=machine, sim_clock=self.env.now,
                                                         episode_step=self.ep_step)
        # print("\t released machine", machine_ident, self.env.now)
        # print("\t releasing order", order_number, self.env.now)
        yield self.order_store.put(order)
        # print("\t released order", order_number, self.env.now)
        self.obs_manager.state_update_on_operation_end(order=order, sim_clock=self.env.now,
                                                       episode_step=self.ep_step)  # important to have updated machine
        # state before the order state update for obs_manger logic

        # ask agent if more orders should be scheduled after processing finished
        self.action_opportunity.succeed()
        self.action_opportunity = self.env.event()

    def check_for_open_tasks(self):
        available_machine_park = self.machine_store.items
        # available_machine_list = [op_capability for machine in available_machine_park for op_capability in
        #                           machine.op_capability]
        if len(available_machine_park) > 0:
            available_order_park = self.order_store.items
            if len(available_order_park) > 0:
                available_machine_types_list = [machine.op_capability for machine in available_machine_park]
                # get waiting orders scheduled on available machine
                required_machines_4_open_orders = [order.machine_type_required for order in available_order_park]
                if len(set(available_machine_types_list) & set(required_machines_4_open_orders)) > 0:
                    return True
        return False

    def init_simpy_env(self, jsp_instance, max_n_machines, max_n_orders):
        self.env = simpy.Environment()
        ## wait for no open actions to run
        self.action_opportunity = self.env.event()
        self.scheduled_operation_events = []
        self.operation_placed_events = []

        self.max_n_orders = max_n_orders
        self.max_n_machines = max_n_machines

        # define machines store and machine object
        self.Machine = namedtuple('Machine', 'number, op_capability, state')
        self.machine_store = simpy.FilterStore(self.env, capacity=self.max_n_machines)

        # define order store and order object
        self.Order = recordtype('Order', 'number, op_durations_lst, op_types_lst, pointer, tot_n_operations, '
                                         'processing_machine, pending_op_duration, machine_type_required, '
                                         'tot_remaining_time, done')
        self.order_store = simpy.FilterStore(self.env, capacity=self.max_n_orders)

        self.machine_store.items = [self.Machine(m, m, -1) for m in range(self.jsp_n_machines)]

        # fill order store
        orders_list = []
        for order_name, order_data in jsp_instance.groupby(['order']):
            orders_list.append(
                copy.deepcopy(
                    self.Order(order_name, order_data['opduration'].values,
                               order_data['machinetype'].apply(lambda x: int(x)).values, 0, order_data.shape[0], -1,
                               order_data['opduration'].iloc[0],
                               order_data['machinetype'].apply(lambda x: int(x)).iloc[0], sum(order_data['opduration']),
                               False)
                )
            )
        self.order_store.items = orders_list

        # filter items in order and machine stores
        self.order_store.items = sorted(self.order_store.items, key=attrgetter('number'))
        self.machine_store.items = sorted(self.machine_store.items, key=attrgetter('number'))

    def reset_jsp(self, jsp_ind):
        if jsp_ind != None:
            self.jsp_ind = jsp_ind
        elif (self.reset_counter % self.repetitions_per_jsp_task == 0):
            if self.random_jsp_iter:
                self.jsp_ind = self.np_random.randint(0, len(self.jsp_lst))
            else:
                self.jsp_ind += 1
                if self.jsp_ind > len(self.jsp_lst) - 1:
                    self.jsp_ind = 0

        self.jsp_task = copy.deepcopy(self.jsp_lst[self.jsp_ind])
        self.jsp_instance = self.jsp_task['jssp_instance']
        self.optimal_time = self.jsp_task['optimal_time']
        self.jsp_id = self.jsp_task['jssp_identification']
        self.jsp_n_jobs = self.jsp_task['n_jobs']
        self.jsp_n_machines = self.jsp_task['n_resources']
        self.n_ops_per_job = self.jsp_task['n_ops_per_job']
        self.max_op_time = self.jsp_task['max_op_time']
        self.max_episode_steps = self.n_ops_per_job * self.jsp_n_jobs

        # print('jsp instance is', self.jsp_instance)

    def reset_jsp_lst(self, jsp_lst):
        self.jsp_lst = jsp_lst

    def set_monitor_mode(self, type):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
