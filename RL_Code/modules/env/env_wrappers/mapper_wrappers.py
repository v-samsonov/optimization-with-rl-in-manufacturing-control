from gym import wrappers

from RL_Code.modules.env.env_wrappers.env_wrappers_action import TransformActionOpRelDuration
from RL_Code.modules.env.env_wrappers.env_wrappers_reward import TransformRewardExp
from RL_Code.modules.env.env_wrappers.r2_wrapper import get_r2_env_wrapper

# Wrappers mapping key wrapper names in run config files to classes. Extend the dict with new wrappers if needed

ENV_WRAPPERS = {
    'OperationRelDurationDiscrete': TransformActionOpRelDuration,
    'Exp_Reward': TransformRewardExp,
    'Gym_Monitor': wrappers.Monitor,
    'Ranked_Reward': get_r2_env_wrapper
}
