import torch

from torchvision.utils import make_grid
from base import BaseTester
from utils import MetricTracker
from tqdm import tqdm


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
            print(len(self.data_loader))
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
