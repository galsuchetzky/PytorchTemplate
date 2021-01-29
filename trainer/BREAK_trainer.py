import numpy as np
import torch

from base_trainer import BaseTrainer
from utils.util import inf_loop, MetricTracker
from tqdm import tqdm
from data_loader.vocabs import batch_to_tensor
from torch.nn import CrossEntropyLoss
from tester.BREAK_tester import Seq2SeqSimpleTester


# TODO add documentation and complete implementation for the Seq2SeqSimpleTrainer
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
        self.vocab = model.vocab
        self.pad_idx = self.vocab['<pad>']
        # TODO the loss function should come from the loss.py file. move it there.
        self.criterion = CrossEntropyLoss(ignore_index=self.pad_idx)
        super().__init__(model,
                         self.criterion,
                         metric_ftns,
                         optimizer,
                         config,
                         device,
                         data_loader,
                         valid_data_loader,
                         lr_scheduler)

        self.question_pad_length = config['data_loader']['question_pad_length']
        self.qdmr_pad_length = config['data_loader']['qdmr_pad_length']
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # Define evaluator.
        self.evaluator = Seq2SeqSimpleTester(self.model,
                                             self.criterion,
                                             self.metric_ftns,
                                             self.config,
                                             self.device,
                                             self.valid_data_loader)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # Sets the model to training mode.
        self.model.train()
        self.train_metrics.reset()
        with tqdm(total=len(self.data_loader)) as progbar:
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
                progbar.update(1)
                epoch_part = str(epoch) + '/' + str(self.epochs)
                progbar.set_postfix(epoch=epoch_part, LOSS=loss.item(),
                                    batch_size=self.data_loader.init_kwargs['batch_size'],
                                    samples=self.data_loader.n_samples)

        # Save the calculated metrics for that epoch.
        log = self.train_metrics.result()

        # If validation split exists, evaluate on validation set as well.
        if self.do_validation:
            # TODO print epoch stuff and add epoch to writer
            # TODO self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            val_log = self.evaluator.test()
            log.update(**{'val_' + k: round(v, 5) for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
