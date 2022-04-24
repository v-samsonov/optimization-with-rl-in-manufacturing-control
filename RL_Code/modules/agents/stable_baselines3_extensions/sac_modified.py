from stable_baselines3 import SAC


class SACModified(SAC):

    def __init__(self, policy, env, learning_rate=0.0003, buffer_size=1000000,
              learning_starts=100, batch_size=256, tau=0.005, gamma=0.99,
              train_freq=1, gradient_steps=1, n_episodes_rollout=- 1, action_noise=None,
              optimize_memory_usage=False, ent_coef='auto', target_update_interval=1,
              target_entropy='auto', use_sde=False, sde_sample_freq=- 1, use_sde_at_warmup=False,
              tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0, seed=None,
              device='auto', _init_setup_model=True, **kwargs):

        SAC.__init__(self, policy, env, learning_rate, buffer_size,
               learning_starts, batch_size, tau, gamma,
               train_freq, gradient_steps, n_episodes_rollout, action_noise,
               optimize_memory_usage, ent_coef, target_update_interval,
               target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup,
               tensorboard_log, create_eval_env, policy_kwargs, verbose, seed,
               device, _init_setup_model)
