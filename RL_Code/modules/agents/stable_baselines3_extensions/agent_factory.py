
from RL_Code.modules.agents.framework_agnostic_extensions.learning_rt_scheduler_constructor import \
    LnRtSchedulerConstrucor

from RL_Code.modules.agents.stable_baselines3_extensions.policy_constructor import assemble_policy_kwargs, build_policy
from RL_Code.modules.agents.stable_baselines3_extensions.noise_factory import ActionNoiseConstructor
from RL_Code.modules.agents.stable_baselines3_extensions.sac_modified import SACModified
from RL_Code.modules.agents.stable_baselines3_extensions.ppo_modified import PPOModified
from RL_Code.modules.agents.stable_baselines3_extensions.dqn_modified import DQNModified
from RL_Code.modules.agents.stable_baselines3_extensions.vectorize_environment import vectorize_env

from RL_Code.modules.agents.agent_factory_abs import AgentConstructorInterface

RL_ALGOS = {
    'stable_baselines3':
        {'sac': SACModified,
         'dqn': DQNModified,
         'ppo': PPOModified
        }
    }


class AgentConstructor(AgentConstructorInterface):

    def build(self, rl_algorithm_tag, environment, rnd_seed,
                                       tebsorboard_path,
                                       **kwargs):
        # Build noise if specified
        action_noise = ActionNoiseConstructor().build(environment=environment, **kwargs)

        policy_kwargs = assemble_policy_kwargs(rl_algorithm_tag, **kwargs)
        policy_type = build_policy(rl_algorithm_tag, **policy_kwargs)
        ln_rt_scheduler = LnRtSchedulerConstrucor().build_ln_rt_scheduler(**kwargs)
        vect_environment = vectorize_env(environment=environment, rnd_seed=rnd_seed, **kwargs)
        if action_noise is None:
            return RL_ALGOS['stable_baselines3'][rl_algorithm_tag](policy=policy_type,
                                                                      env=vect_environment,
                                                                      learning_rate=ln_rt_scheduler.value,
                                                                      seed=rnd_seed,
                                                                      tensorboard_log=tebsorboard_path,
                                                                      **kwargs)
        else:
            return RL_ALGOS['stable_baselines3'][rl_algorithm_tag](policy=policy_type,
                                                                      env=vect_environment,
                                                                      learning_rate=ln_rt_scheduler.value,
                                                                      action_noise=action_noise,
                                                                      seed=rnd_seed,
                                                                      tensorboard_log=tebsorboard_path,
                                                                      **kwargs)
