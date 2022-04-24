import gym
import numpy as np

from RL_Code.modules.env.env_wrapper_factory import EnvWrapperConstructor


class EnvironmentConstructor():
    # Client Component
    def build(self, env_tag, rand_seed, ranked_reward=False, eval_flag=False, *args, **kwargs):
        environment = self._get_builder(env_tag)
        return environment(rand_seed=rand_seed, env_tag=env_tag, ranked_reward=ranked_reward, eval_flag=eval_flag,
                           *args, **kwargs)

    # Creator Component
    def _get_builder(self, env_tag):
        if env_tag == 'jsp_simpy':
            return self._build_simpy_jsp_env
        if env_tag == 'jsp_l2d':
            return self._build_l2d_jsp_env
        else:
            return self._build_other_gym_env

    # Product Components
    def _build_simpy_jsp_env(self, rand_seed, wrappers_lst, envs_number, eval_flag, *args, **kwargs):
        from RL_Code.modules.env.simpy_jsp_env.simpy_env import JobShop
        environment_lst = []
        for i in range(envs_number):
            env = JobShop(**kwargs)
            env.seed(rand_seed + i)
            env_wrapper_object = EnvWrapperConstructor()
            for wrapper_tag, pars in wrappers_lst.items():
                env = env_wrapper_object.build(environment=env, wrapper_tag=wrapper_tag,
                                               *args, **{**pars, **kwargs}
                                               )
            environment_lst.append(env)
        if eval_flag:
            return env
        else:
            return environment_lst

    def _build_l2d_jsp_env(self, env_tag, rand_seed, wrappers_lst, num_envs, jsp_lst, eval_flag, *args, **kwargs):
        from RL_Code.modules.env.l2d_jsp_env.JSSP_Env import SJSSP
        from RL_Code.modules.env.l2d_jsp_env.jsp_generator_constructor import JSP_INST_GEN
        if eval_flag:
            environment = SJSSP(jsp_lst=jsp_lst, seed=rand_seed, instance_gen=JSP_INST_GEN[kwargs['jsp_instance_gen']],
                                *args, **kwargs)
            env_wrapper_object = EnvWrapperConstructor()
            for wrapper_tag in wrappers_lst:
                environment = env_wrapper_object.build(environment=environment, wrapper_tag=wrapper_tag, *args,
                                                       **kwargs)
        else:
            environment = []
            ss = np.random.SeedSequence(rand_seed)
            child_seeds = ss.spawn(num_envs)
            for i in range(num_envs):
                env = SJSSP(jsp_lst=jsp_lst, seed=child_seeds[i], instance_gen=JSP_INST_GEN[kwargs['jsp_instance_gen']],
                            *args, **kwargs)
                for wrapper_tag in wrappers_lst:
                    env = EnvWrapperConstructor().build(environment=env, wrapper_tag=wrapper_tag, *args,
                                                        **kwargs)
                environment.append(env)
        return environment

    def _build_other_gym_env(self, env_tag, rand_seed, wrappers_lst, envs_number, eval_flag, *args, **kwargs):
        environment_lst = []
        for i in range(envs_number):
            env = gym.make(env_tag)
            env.seed(rand_seed)
            env_wrapper_object = EnvWrapperConstructor()
            for wrapper_tag in wrappers_lst:
                env = env_wrapper_object.build(environment=env, wrapper_tag=wrapper_tag, **kwargs)
            environment_lst.append(env)
        if eval_flag:
            return env
        else:
            return environment_lst
