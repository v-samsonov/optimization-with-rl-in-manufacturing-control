from abc import ABC, abstractmethod


class AgentManagerInterface(ABC):

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def retrain(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass
