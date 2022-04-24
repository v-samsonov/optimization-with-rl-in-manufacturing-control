
# from RL_Code.modules.agents.l2d_agent_extensions.policy_constructor import assemble_policy_kwargs, build_policy
from RL_Code.modules.agents.l2d_agent_extensions.ppo_modified import PPOModified
# from RL_Code.modules.agents.l2d_agent_extensions.l2d.PPO_jssp_multiInstances import PPO
# from RL_Code.modules.agents.l2d_agent_extensions.vectorize_environment import vectorize_env

from RL_Code.modules.agents.agent_factory_abs import AgentConstructorInterface

RL_ALGOS = {
    'l2d':
        {'ppo': PPOModified,
        }
    }


class AgentConstructor(AgentConstructorInterface):

    def build(self, rl_algorithm_tag, environment, rnd_seed, tebsorboard_path, **kwargs):

        #policy_kwargs = assemble_policy_kwargs(rl_algorithm_tag, **kwargs)
        #policy_type = build_policy(rl_algorithm_tag, **policy_kwargs)
        #vect_environment = vectorize_env(environment=environment, rnd_seed=rnd_seed, **kwargs)

        return RL_ALGOS['l2d'][rl_algorithm_tag](#policy=policy_type,
                                                                  env=environment,
                                                                  seed=rnd_seed,
                                                                  tensorboard_log=tebsorboard_path,
                                                                  **kwargs)
