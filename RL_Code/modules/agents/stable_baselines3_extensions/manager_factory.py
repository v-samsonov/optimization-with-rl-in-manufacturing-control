from RL_Code.modules.agents.stable_baselines3_extensions.agent_factory import RL_ALGOS
from stable_baselines3.common.callbacks import CheckpointCallback as CheckpointCallback3
from RL_Code.modules.agents.manager_factory_abs import AgentManagerInterface


class AgentManager(AgentManagerInterface):

    def train(self, rl_agent, training_steps, tb_log_name, n_checkpoints,
                                   agent_path, agent_name, log_interval, **kwargs):
        save_freq = int(training_steps / n_checkpoints)
        checkpoint_callback = CheckpointCallback3(save_freq=save_freq, save_path=agent_path,
                                                  name_prefix=agent_name)
        rl_agent.learn(total_timesteps=training_steps, log_interval=log_interval, tb_log_name=tb_log_name,
                       # reset_num_timesteps=False,
                       callback=checkpoint_callback)

        environment = rl_agent.get_env()
        return rl_agent, environment

    def retrain(self, rl_agent, training_steps, environment, tb_log_name, n_checkpoints,
                                    agent_path, agent_name, log_interval, **kwargs):
        rl_agent.set_env(environment)
        save_freq = int(training_steps / n_checkpoints)
        checkpoint_callback = CheckpointCallback3(save_freq=save_freq, save_path=agent_path,
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

    def step(self, rl_agent, obs, **kwargs):
        action, _states = rl_agent.predict(obs, deterministic=True)
        return action
