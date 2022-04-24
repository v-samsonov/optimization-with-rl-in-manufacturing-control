from RL_Code.modules.agents.framework_factory_abs import FrameworkConstructorInterface

from RL_Code.modules.agents.stable_baselines2_action_filtering.agent_factory import AgentConstructor
from RL_Code.modules.agents.stable_baselines2_action_filtering.manager_factory import AgentManager


class StableBaselines2Constructor(FrameworkConstructorInterface):

    def __init__(self):
        self.agent_factory = AgentConstructor
        self.agent_manager_factory = AgentManager

    def agent_factory(self):
        return self.agent_factory

    def agent_manager_factory(self):
        return self.agent_manager_factory
