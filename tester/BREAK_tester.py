import torch
from tqdm import tqdm

from base import BaseTester
from utils import MetricTracker
from data_loader import batch_to_tensor
from torch.nn import CrossEntropyLoss


# TODO add documentation and complete implementation for the Seq2SeqSimpleTrainer


class Seq2SeqSimpleTester(BaseTester):
    """
    Trainer for a simple seq2seq mode.
    """

    def __init__(self, model, criterion, metric_ftns, config, device,
                 data_loader, evaluation=True):
        """

        :param model: A model to test.
        :param criterion: we ignore this value and overwrite it
        :param metric_ftns: The names of the metric functions to use.
        :param config: The configuration.
        :param device: The device to use for the testing.
        :param data_loader: The dataloader to use for loading the testing data.
        """
        self.vocab = model.vocab
        self.question_pad_length = config['data_loader']['question_pad_length']
        self.qdmr_pad_length = config['data_loader']['qdmr_pad_length']
        self.pad_idx = self.vocab['<pad>']

        # Overriding the criterion.
        self.criterion = CrossEntropyLoss(ignore_index=self.pad_idx)

        super().__init__(model, self.criterion, metric_ftns, config, device, data_loader, evaluation)

        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _evaluate(self):
        """
        Validate after training an epoch.

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        # Sets the model to evaluation mode.
        self.valid_metrics.reset()
        with tqdm(total=len(self.data_loader)) as progbar:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, mask_data = batch_to_tensor(self.vocab, data, self.question_pad_length, self.device)
                target, mask_target = batch_to_tensor(self.vocab, target, self.qdmr_pad_length, self.device)

                # Run the model on the batch and calculate the loss
                output = self.model(data, target, evaluation_mode=True)
                output = torch.transpose(output, 1, 2)
                pred = torch.argmax(output, dim=1)
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred, target))

                # Update the progress bar.
                progbar.update(1)
                progbar.set_postfix(LOSS=loss.item(),
                                    batch_size=self.data_loader.init_kwargs['batch_size'],
                                    samples=self.data_loader.n_samples)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
