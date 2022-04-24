from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.deepq.policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.deepq.build_graph import build_act, build_train  # noqa
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.deepq.dqn import DQN
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa


def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN
    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)