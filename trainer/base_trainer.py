import torch
import time

from abc import abstractmethod
from numpy import inf
from logger.visualization import TensorboardWriter
from utils.util import time_elapsed, time_remaining


class BaseTrainer:
    """
    Base class for all trainers.
    Handles the training loop, logging, saving checkpoints, timings and presenting the training results.
    """

    def __init__(self, model, criterion, train_metric_ftns, eval_metric_ftns, optimizer, config, device, data_loader,
                 valid_data_loader, lr_scheduler):
        """
        Initiates the Base trainer.
        :param model:               The model to train.
        :param criterion:           The loss function.
        :param train_metric_ftns:   The metrics on which the model will be evaluated during evaluation or train time.
        :param eval_metric_ftns:    The metrics on which the model will be evaluated during evaluation or train time.
        :param optimizer:           The optimizer to use for optimizing the parameters of the model.
        :param config:              Configuration file.
        :param device:              The device to use for computations.
        :param data_loader:         Dataloader for the train dataset.
        :param valid_data_loader:   Dataloader for the validation dataset.
        :param lr_scheduler:        Scheduler for the learning rate.
        """
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.train_metric_ftns = train_metric_ftns
        self.eval_metric_ftns = eval_metric_ftns
        self.optimizer = optimizer
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        # Get training configurations
        cfg_trainer = config['trainer']

        # Metrics to display for the best model.
        self.best_model_metrics_log = cfg_trainer['best_model_metrics_log'].split()
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']  # Once in how many epochs to save the models parameters.

        # Metric which will be used to choose the best model
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best model.
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max'], "Invalid monitor mode, should be min or max"

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # Dictionary to keep the metrics results of the best model.
        self.model_best_metrics = {}

        # The epoch to start working from.
        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        print(self.checkpoint_dir)

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        # If resume path is given, resume training from checkpoint.
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        :param epoch: Current epoch number.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic.
        """
        # Log start and set timer for timing the train.
        self.logger.info('Starting training...')
        train_start_time = time.time()

        not_improved_count = 0  # Counts the amount of epochs where no new best model is found.
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Train for one epoch and time it.
            epoch_start_time = time.time()
            result = self._train_epoch(epoch)
            result['epoch time'] = time_elapsed(epoch_start_time)
            result['Time elapsed'] = time_elapsed(train_start_time)
            result['Time remaining'] = time_remaining(train_start_time, epoch / self.epochs)

            # save logged information into log dict.
            log = {'epoch': epoch}
            log.update(result)

            # evaluate model performance according to configured metric, save best checkpoint as model_best.
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                # If the model improved, save the best monitored result, the whole log and save the model as best.
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.model_best_metrics = log
                    not_improved_count = 0
                    best = True
                    self.logger.info("BEST MODEL FOUND, SAVING")
                    self._save_checkpoint(epoch, save_best=best)
                else:
                    not_improved_count += 1

                # If the model has not improved for self.early_stop epochs, stop the training.
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            # Save the model periodically.
            if epoch % self.save_period == 0:
                self.logger.info("Saving current model (save period)...")
                self._save_checkpoint(epoch)

            # print logged information to the screen of current and best epochs.
            self.logger.info('------------ Current ------------')
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if self.model_best_metrics:
                self.logger.info('------------- Best --------------')
                for key in self.best_model_metrics_log:
                    self.logger.info('    {:15s}: {}'.format(str(key), self.model_best_metrics[key]))

        self.logger.info(f'Training finished.\nTotal training time: {time_elapsed(train_start_time)}')
        self.logger.info(f'best model accuracy: {round(self.mnt_best, 3) * 100}%')

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints.

        :param epoch: current epoch number.
        :param log: logging information of the epoch.
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'.
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints.
        :param resume_path: Checkpoint path to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
