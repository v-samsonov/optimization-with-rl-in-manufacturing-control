from RL_Code.modules.agents.framework_factory_abs import FrameworkConstructorInterface

from RL_Code.modules.agents.l2d_agent_extensions.agent_factory import AgentConstructor
from RL_Code.modules.agents.l2d_agent_extensions.manager_factory import AgentManager


class L2DConstructor(FrameworkConstructorInterface):

    def __init__(self):
        self.agent_factory = AgentConstructor
        self.agent_manager_factory = AgentManager

    def agent_factory(self):
        return self.agent_factory

    def agent_manager_factory(self):
        return self.agent_manager_factory
