import numpy as np
import torch
from parse_config import ConfigParser

from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from data_loader import batch_to_tensor
from torch.nn import CrossEntropyLoss


# TODO add documentation and complete implementation for the Seq2SeqSimpleTrainer
class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()
        with tqdm(total=self.data_loader.n_samples) as progbar:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, target))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
                progbar.update(self.data_loader.init_kwargs['batch_size'])
                epoch_part = str(epoch) + '/' + str(self.epochs)
                progbar.set_postfix(epoch=epoch_part, NLL=loss.item())
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: round(v, 5) for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Seq2SeqSimpleTrainer(BaseTrainer):
    """
    Trainer for a simple seq2seq mode.
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        """

        :param model:
        :param criterion: we ignore this value and overwrite it
        :param metric_ftns:
        :param optimizer:
        :param config:
        :param device:
        :param data_loader:
        :param valid_data_loader:
        :param lr_scheduler:
        :param len_epoch:
        """
        # TODO document this
        self.config = config
        self.device = device
        self.vocab = model.vocab
        self.data_loader = data_loader
        self.question_pad_length = config['data_loader']['question_pad_length']
        self.qdmr_pad_length = config['data_loader']['qdmr_pad_length']
        self.pad_idx = self.vocab['<pad>']
        self.criterion = CrossEntropyLoss(ignore_index=self.pad_idx)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        super().__init__(model, self.criterion, metric_ftns, optimizer, config)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # Sets the model to training mode.
        self.model.train()
        self.train_metrics.reset()
        with tqdm(total=self.data_loader.n_samples) as progbar:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, mask_data = batch_to_tensor(self.vocab, data, self.question_pad_length, self.device)
                target, mask_target = batch_to_tensor(self.vocab, target, self.qdmr_pad_length, self.device)
                # Run the model on the batch
                self.optimizer.zero_grad()
                output = self.model(data, target)

                # loss expects (minibatch, classes, seq_len)
                # out now is (batch_size, seq_len, output_size)
                # out after transpose is (batch_size, output_size, seq_len)
                output = torch.transpose(output, 1, 2)

                # Calculate the loss and perform optimization step.
                # TODO test properly use of masks
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

                # Update metrics
                self.train_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    with torch.no_grad():
                        pred = torch.argmax(output, dim=1)
                    self.train_metrics.update(met.__name__, met(pred, target))

                # Log progress
                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))
                    # TODO set this to write the text examples
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break

                # Update the progress bar.
                progbar.update(self.data_loader.init_kwargs['batch_size'])
                epoch_part = str(epoch) + '/' + str(self.epochs)
                progbar.set_postfix(epoch=epoch_part, LOSS=loss.item())

        # Save the calculated metrics for that epoch.
        log = self.train_metrics.result()

        # If validation split exists, evaluate on validation set as well.
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: round(v, 5) for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch.

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        # Sets the model to evaluation mode.
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, mask_data = batch_to_tensor(self.vocab, data, self.question_pad_length, self.device)
                target, mask_target = batch_to_tensor(self.vocab, target, self.qdmr_pad_length, self.device)

                # Run the model on the batch and calculate the loss
                output = self.model(data, target, evaluation_mode=True)
                output = torch.transpose(output, 1, 2)
                pred = torch.argmax(output, dim=1)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                # print('----------------------------------------------------------------')
                # print(tensor_to_str(self.vocab, pred))
                # print(tensor_to_str(self.vocab, target))
                # print('----------------------------------------------------------------')

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
