import torch
import time

from abc import abstractmethod
from logger import TensorboardWriter
from utils import time_elapsed


class BaseTester:
    """
    Base class for all testers.
    Handles the testing loop, logging, loading checkpoints, timings and presenting the testing results.
    """

    def __init__(self, model, criterion, metric_ftns, config, device, data_loader, evaluation=True):
        """
        Initiates the Base tester.
        :param model:       The model to test.
        :param criterion:   The loss function.
        :param metric_ftns: The metrics on which the model will be evaluated during test time.
        :param config:      Configuration file.
        :param device:      The device to use for the computations.
        :param data_loader: Dataloader for the dataset.
        :param evaluation:  True if the tester is used as evaluator while training, False if used for testing the model.
        """
        self.config = config
        self.logger = config.get_logger('tester', config['tester']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.device = device
        self.data_loader = data_loader
        self.evaluation = evaluation

        # Get testing configurations
        cfg_tester = config['tester']

        # Metrics to display for the model.
        self.metrics_log = cfg_tester['metrics_log'].split()

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_tester['tensorboard'])

    @abstractmethod
    def _evaluate(self):
        """
        Evaluation logic for a specific model.
        Can be used for evaluation or testing.
        """
        raise NotImplementedError

    def test(self):
        """
        Common testing operations.
        """
        # Log start and set timer for timing the train.
        # TODO uncomment this
        # self.logger.info('Starting evaluation on ' + self.data_loader.get_dataset_type() + ' set')
        eval_start_time = time.time()

        # Sets the model to evaluation mode.
        self.model.eval()

        with torch.no_grad():

            result = self._evaluate()

            # save logged information into log dict.
            log = {}
            log.update(result)

            # When testing, print test results.
            if not self.evaluation:
                self.logger.info('Testing Finished.')
                self.logger.info('------------ Test Result ------------')
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

                self.logger.info(f'Total Test time: {time_elapsed(eval_start_time)}')
                self.logger.info(f'\nTo view results on tensorboard, run: tensorboard --logdir saved/log/')
        return log
