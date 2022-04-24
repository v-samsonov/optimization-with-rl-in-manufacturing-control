import collections

import numpy as np
from gym import ObservationWrapper
from gym.spaces.box import Box


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = Box(-np.inf, np.inf, (121,), np.float32)

    def observation(self, observation):
        # flatten dict and extract values
        state_all_lst = self._hierarchical_dict_to_list(observation)
        state_all_arr = np.concatenate(state_all_lst, axis=0)
        state_all_arr = np.where(state_all_arr < 0, -1, state_all_arr)
        return state_all_arr.astype(np.float32)

    def _hierarchical_dict_to_list(self, d, parent_key=''):
        obs_list = []
        for k, v in d.items():
            new_key = k
            if isinstance(v, collections.MutableMapping):
                obs_list.extend(self._hierarchical_dict_to_list(v, new_key))
            else:
                obs_list.append(v.flatten())
        return obs_list


class L2DObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(L2DObservation, self).__init__(env)

    def observation(self, observation):
        adj = observation['l2d']['adj'].astype(np.float32)
        fea = observation['l2d']['fea'].astype(np.float32)
        candidate = observation['l2d']['candidate'].astype(np.int64)
        mask = observation['l2d']['mask'].astype(np.bool)
        return {'adj': adj, 'fea': fea, 'omega': candidate, 'mask': mask}
