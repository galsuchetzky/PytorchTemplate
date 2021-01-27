import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self, device):
        """
        Initiates the base model.
        """
        self.device = device

        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :param inputs: the inputs to the forward pass.
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
