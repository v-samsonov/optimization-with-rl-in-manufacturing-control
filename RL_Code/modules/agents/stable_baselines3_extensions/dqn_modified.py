from stable_baselines3 import DQN


class DQNModified(DQN):

    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
             exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True,
             learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
             prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
             prioritized_replay_eps=1e-6, param_noise=False,
             n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
             _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None,  **kwargs):

        DQN.__init__(self, policy, env, gamma, learning_rate, buffer_size, exploration_fraction,
             exploration_final_eps, exploration_initial_eps, train_freq, batch_size, double_q,
             learning_starts, target_network_update_freq, prioritized_replay,
             prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_beta_iters,
             prioritized_replay_eps, param_noise,
             n_cpu_tf_sess, verbose, tensorboard_log,
             _init_setup_model, policy_kwargs, full_tensorboard_log, seed)
