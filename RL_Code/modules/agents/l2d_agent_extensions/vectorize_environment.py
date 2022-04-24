from RL_Code.modules.env.l2d_jsp_env.JSSP_Env import SJSSP
import torch
import numpy as np


def vectorize_env(environment, envs_number, rnd_seed=0, env_wrapper='DummyVecEnv', **kwargs):

    def _init():

        return environment

    return _init()
