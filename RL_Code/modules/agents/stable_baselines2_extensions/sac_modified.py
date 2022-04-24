from stable_baselines import SAC


class SACModified(SAC):

    def __init__(self, policy, env, gamma=0.99, learning_rate=0.0003, buffer_size=50000, learning_starts=100,
                 train_freq=1, batch_size=64, tau=0.005, ent_coef='auto', target_update_interval=1, gradient_steps=1,
                 target_entropy='auto', action_noise=None, random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None,
                 n_cpu_tf_sess=None, **kwargs):
        SAC.__init__(self, policy, env, gamma, learning_rate, buffer_size,
                     learning_starts, train_freq, batch_size,
                     tau, ent_coef, target_update_interval,
                     gradient_steps, target_entropy, action_noise,
                     random_exploration, verbose, tensorboard_log,
                     _init_setup_model, policy_kwargs, full_tensorboard_log, seed, n_cpu_tf_sess)
