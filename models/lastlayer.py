from abc import ABC, abstractmethod
import torch.nn as nn

class LastLayer(ABC):
    @abstractmethod
    def last(self) -> nn.Module:
        """Return the last layer of the model."""
        pass
