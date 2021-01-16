import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        """
        Initiates the base model.
        """
        self.device = None

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

    def set_device(self, device):
        """
        Sets the device of the model.
        :param device: The device to use.
        """
        # TODO maybe get the device from the init, requires a change in the config realization.
        self.device = device
