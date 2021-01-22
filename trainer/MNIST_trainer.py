import numpy as np

from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from tester import MNISTTester


class MNISTTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_fns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_fns, optimizer, config)
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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        # Define evaluator.
        self.evaluator = MNISTTester(self.model,
                                     self.criterion,
                                     self.metric_fns,
                                     self.config,
                                     self.device,
                                     self.valid_data_loader)

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
                for met in self.metric_fns:
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
