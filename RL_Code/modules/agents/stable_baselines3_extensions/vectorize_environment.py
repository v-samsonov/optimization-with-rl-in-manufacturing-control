from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

ENV_WRAPPERS = {
    'DummyVecEnv': DummyVecEnv,
    'SubprocVecEnv': SubprocVecEnv
}


def vectorize_env(environment, envs_number, rnd_seed=0, env_wrapper='DummyVecEnv', **kwargs):
    """
    Utility function for stable baseline multiprocess env.
    :param env_: (object) openai gym env
    :param envs_number: (int) the number of environments to have in subprocesses
    :param rnd_seed: (int) the initial random seed
    """

    def env_set_rnd_seed(environment, rank, rnd_seed):
        """
        :param rank: (int) index of the subprocess
        """
        if rnd_seed is not None:
            environment.seed(rnd_seed + rank)
            met_exists_flag = False
        if "logger" in dir(environment):
            environment.logger.reset_log_path(rank)

        return environment

    def _init():
        if env_wrapper in ENV_WRAPPERS:
            envs = ENV_WRAPPERS[env_wrapper]([lambda e=e:e for e in environment])
            # envs = DummyVecEnv([lambda: eval_env])
            # set_global_seeds(rnd_seed)
        else:
            raise ValueError(f'{env_wrapper} is an undefined environment vectorizer')
        return envs

    #set_global_seeds(rnd_seed)
    return _init()
