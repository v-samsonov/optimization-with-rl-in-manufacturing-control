from abc import ABC, abstractmethod


class ObservationManagerInterface(ABC):

    @abstractmethod
    def reset_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_observations(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_optimal_time(self, *args, **kwargs):
        pass
