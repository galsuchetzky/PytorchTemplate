import numpy as np
import torch
from parse_config import ConfigParser

from torchvision.utils import make_grid
from base import BaseTester
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from data_loader import batch_to_tensor
from torch.nn import CrossEntropyLoss


# # TODO add documentation and complete implementation for the Seq2SeqSimpleTrainer
class MNISTTester(BaseTester):
    """
    Trainer for a simple seq2seq mode.
    """

    def __init__(self, model, criterion, metric_fns, config, device,
                 data_loader, evaluation=True):
        """

        :param model:
        :param criterion: we ignore this value and overwrite it
        :param metric_fns:
        :param optimizer:
        :param config:
        :param device:
        :param data_loader:
        :param valid_data_loader:
        :param lr_scheduler:
        :param len_epoch:
        """

        super().__init__(model, criterion, metric_fns, config, device, data_loader, evaluation)

        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _evaluate(self):
        """
        Validate after training an epoch.

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        # Sets the model to evaluation mode.
        self.valid_metrics.reset()
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))

        for i, (data, target) in enumerate(tqdm(self.data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)

            # computing loss, metrics on test set
            loss = self.criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(self.metric_ftns):
                total_metrics[i] += metric(output, target) * batch_size

            self.valid_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(output, target))
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


class Seq2SeqSimpleTester(BaseTester):
    """
    Trainer for a simple seq2seq mode.
    """

    def __init__(self, model, criterion, metric_fns, config, device,
                 data_loader, evaluation=True):
        """

        :param model:
        :param criterion: we ignore this value and overwrite it
        :param metric_fns:
        :param optimizer:
        :param config:
        :param device:
        :param data_loader:
        :param valid_data_loader:
        :param lr_scheduler:
        :param len_epoch:
        """
        # TODO document this
        self.vocab = model.vocab
        self.question_pad_length = config['data_loader']['question_pad_length']
        self.qdmr_pad_length = config['data_loader']['qdmr_pad_length']
        self.pad_idx = self.vocab['<pad>']

        # Overriding the criterion.
        self.criterion = CrossEntropyLoss(ignore_index=self.pad_idx)

        super().__init__(model, self.criterion, metric_fns, config, device, data_loader, evaluation)

        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)

    def _evaluate(self):
        """
        Validate after training an epoch.

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        # Sets the model to evaluation mode.
        self.valid_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, mask_data = batch_to_tensor(self.vocab, data, self.question_pad_length, self.device)
            target, mask_target = batch_to_tensor(self.vocab, target, self.qdmr_pad_length, self.device)

            # Run the model on the batch and calculate the loss
            output = self.model(data, target, evaluation_mode=True)
            output = torch.transpose(output, 1, 2)
            pred = torch.argmax(output, dim=1)
            loss = self.criterion(output, target)

            self.valid_metrics.update('loss', loss.item())
            for met in self.metric_fns:
                self.valid_metrics.update(met.__name__, met(pred, target))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
