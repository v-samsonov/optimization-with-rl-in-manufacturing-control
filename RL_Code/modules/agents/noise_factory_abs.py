from abc import ABC, abstractmethod


class ActionNoiseConstructorInterface(ABC):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def _get_noise_constructor(self):
        pass

    @abstractmethod
    def no_noise(self):
        pass
