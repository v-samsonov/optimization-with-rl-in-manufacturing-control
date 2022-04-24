from stable_baselines.common.callbacks import BaseCallback, CallbackList

from RL_Code.modules.agents.manager_factory_abs import AgentManagerInterface
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.mod_callbacs import ModCheckpointCallback
from RL_Code.modules.agents.stable_baselines2_action_filtering.agent_factory import RL_ALGOS


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_step(self):
        """
        This method is called before the first rollout starts.
        """
        # pass


class AgentManager(AgentManagerInterface):

    # Product Components for Stable Baselines
    def train(self, rl_agent, training_steps, tb_log_name, n_checkpoints,
              agent_path, agent_name, log_interval, **kwargs):
        save_freq = int(training_steps / n_checkpoints)
        checkpoint_callback = ModCheckpointCallback(save_freq=save_freq, save_path=agent_path,
                                                 name_prefix=agent_name)
        wandb_callback = WandbCallback()
        callbacks = CallbackList([checkpoint_callback, wandb_callback])
        rl_agent.learn(total_timesteps=training_steps, log_interval=log_interval, tb_log_name=tb_log_name,
                       # reset_num_timesteps=False,
                       callback=callbacks)
        environment = rl_agent.get_env()
        return rl_agent, environment

    def retrain(self, rl_agent, training_steps, environment, tb_log_name, n_checkpoints,
                agent_path, agent_name, log_interval, **kwargs):
        rl_agent.set_env(environment)
        save_freq = int(training_steps / n_checkpoints)
        checkpoint_callback = ModCheckpointCallback(save_freq=save_freq, save_path=agent_path,
                                                 name_prefix=agent_name)
        rl_agent.learn(total_timesteps=training_steps, log_interval=log_interval, tb_log_name=tb_log_name,
                       reset_num_timesteps=False,
                       callback=checkpoint_callback)
        environment = rl_agent.get_env()
        return rl_agent, environment

    def save(self, rl_agent, agent_path='', **kwargs):
        rl_agent.save(agent_path)

    def load(self, rl_framework_tag, rl_algorithm_tag, agent_path='', **kwargs):
        rl_agent = RL_ALGOS[rl_framework_tag][rl_algorithm_tag].load(agent_path)
        return rl_agent

    def step(self, rl_agent, obs, action_mask=None, **kwargs):
        action, _states = rl_agent.predict(observation=obs, action_mask=action_mask, deterministic=True)
        return action
