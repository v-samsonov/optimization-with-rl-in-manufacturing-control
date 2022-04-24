import gym
import numpy as np
from gym.utils import EzPickle

from RL_Code.modules.env.l2d_jsp_env.permissibleLS import permissibleLeftShift
from RL_Code.modules.env.l2d_jsp_env.uniform_instance_gen import override
from RL_Code.modules.env.l2d_jsp_env.updateAdjMat import getActionNbghs
from RL_Code.modules.env.l2d_jsp_env.updateEntTimeLB import calEndTimeLB


# from RL_Code.modules.env.l2d_jsp_env.uniform_instance_gen import uni_instance_gen


class SJSSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m,
                 jsp_lst,
                 high,
                 low,
                 et_normalize_coef,
                 init_quality_flag,
                 rewardscale,
                 seed,
                 instance_gen,
                 **kwargs):
        EzPickle.__init__(self)
        self.jsp_lst = jsp_lst
        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        self.low = low
        self.high = high
        self.et_normalize_coef = et_normalize_coef
        self.init_quality_flag = init_quality_flag
        self.rewardscale = rewardscale
        self.sum_of_rewards = 0
        self.rng = np.random.default_rng(seed)
        self.instance_gen = instance_gen

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect
        if action not in self.partial_sol_sequeence:

            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m,
                                                     mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs,
                                                     high=self.high)
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + dur_a

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # adj matrix
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1

        # prepare for return
        fea = np.concatenate((self.LBs.reshape(-1, 1) / self.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        reward = - (self.LBs.max() - self.max_endTime)

        if reward == 0:
            reward = self.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        self.sum_of_rewards -= reward

        # print(str(self.initQuality + self.sum_of_rewards) + "   " + str(self.max_endTime))

        # print("step: ", self.step_count, "reward: ", reward)
        return {'adj': self.adj, 'fea': fea, 'omega': self.omega, 'mask': self.mask}, reward, self.done(), {
            'makespan': self.max_endTime}

    @override
    def reset(self, jsp_ind):


        if jsp_ind is None:
            if self.instance_gen is None: # if no instance generator is created pick randomly on isntance from the training list
                data = self.jsp_lst[self.random_generator.integers(low=0, high=self.jsp_lst, size=1)[0]]
            else: # generate one instance
                data = self.instance_gen(n_j=self.number_of_jobs, n_m=self.number_of_machines, low=self.low,
                                         high=self.high, random_generator=self.rng, n_ops_per_job=self.number_of_jobs)
        else: # pick chosen instance from the training list
            data = (self.jsp_lst[jsp_ind]['jssp_instance']["durations"],
                    self.jsp_lst[jsp_ind]['jssp_instance']["machines"])

        # if jsp_ind == -1:
        #     data = self.jsp_lst[self.random_generator.integers(low=0, high=self.jsp_lst, size=1)[0]]
        # elif jsp_ind == -2:
        #     data = self.instance_gen(n_j=self.number_of_jobs, n_m=self.number_of_machines, low=self.low,
        #                              high=self.high, random_generator=self.rng, n_ops_per_job=self.number_of_jobs, )
        # else:
        #     data = (self.jsp_lst[jsp_ind]['jssp_instance']["durations"],
        #             self.jsp_lst[jsp_ind]['jssp_instance']["machines"])

        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0
        self.sum_of_rewards = 0

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # initialize features
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not self.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1) / self.et_normalize_coef,
                              # self.dur.reshape(-1, 1)/self.high,
                              # wkr.reshape(-1, 1)/self.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)

        # initialize mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # start time of operations on machines
        self.mchsStartTimes = -self.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)

        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        return {'adj': self.adj, 'fea': fea, 'omega': self.omega, 'mask': self.mask}

    def set_monitor_mode(self, type):
        pass
