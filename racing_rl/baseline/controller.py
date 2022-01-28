from abc import abstractmethod
from typing import Dict


class Controller():
    @abstractmethod
    def predict(self, *args, **kwargs) -> Dict[str, float]:
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass