from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.a2c import A2C
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.acer import ACER
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.acktr import ACKTR
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.deepq import DQN
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.her import HER
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.ppo2 import PPO2
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.td3 import TD3
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.sac import SAC

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.ddpg import DDPG
    from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.gail import GAIL
    from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.ppo1 import PPO1
    from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.trpo_mpi import TRPO
del mpi4py

__version__ = "2.9.0a0"
