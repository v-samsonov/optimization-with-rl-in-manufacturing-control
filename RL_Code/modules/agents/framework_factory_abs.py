from abc import ABC, abstractmethod


class FrameworkConstructorInterface(ABC):

    @property
    @abstractmethod
    def agent_factory(self):
        pass

    @property
    @abstractmethod
    def agent_manager_factory(self):
        pass
