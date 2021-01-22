import torch
from tqdm import tqdm

from base import BaseTester
from utils import MetricTracker
from data_loader import batch_to_tensor
from torch.nn import CrossEntropyLoss


# # TODO add documentation and complete implementation for the Seq2SeqSimpleTrainer


class Seq2SeqSimpleTester(BaseTester):
    """
    Trainer for a simple seq2seq mode.
    """

    def __init__(self, model, criterion, metric_ftns, config, device,
                 data_loader, evaluation=True):
        """

        :param model:
        :param criterion: we ignore this value and overwrite it
        :param metric_ftns:
        :param config:
        :param device:
        :param data_loader:
        """
        # TODO document this
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
        print("dataloader len is", len(self.data_loader)) # TODO for debug should be 300, yet it is 241 for some reason.
        for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
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

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
