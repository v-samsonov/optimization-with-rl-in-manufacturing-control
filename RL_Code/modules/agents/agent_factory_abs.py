from abc import ABC, abstractmethod


class AgentConstructorInterface(ABC):

    @abstractmethod
    def build(self, *args, **kwargs):
        pass
