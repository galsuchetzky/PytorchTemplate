import torch
import random
import time
import pandas as pd

from tqdm import tqdm
from .base_tester import BaseTester
from utils.util import MetricTracker
from data_loader.vocabs import batch_to_tensor, batch_to_str, pred_batch_to_str, tokenize_lexicon_str
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
        # TODO add logger and log "starting evaluation"

        self.vocab = model.vocab
        self.question_pad_length = config['data_loader']['question_pad_length']
        self.qdmr_pad_length = config['data_loader']['qdmr_pad_length']
        self.lexicon_pad_length = config['data_loader']['lexicon_pad_length']
        self.pad_idx = self.vocab['<pad>']

        # Overriding the criterion.
        # self.criterion = CrossEntropyLoss(ignore_index=self.pad_idx)
        self.criterion = criterion
        super().__init__(model, self.criterion, metric_ftns, config, device, data_loader, evaluation)

        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _evaluate(self):
        """
        Validate after training an epoch.
        Used with  gold target

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        # Choose 2 random examples from the dev set and print their prediction.
        batch_index1 = random.randint(0, len(self.data_loader) - 1) - 1
        example_index1 = random.randint(0, self.data_loader.batch_size - 1)
        batch_index2 = random.randint(0, len(self.data_loader) - 1) - 1
        example_index2 = random.randint(0, self.data_loader.batch_size - 1)
        questions = []
        decompositions = []
        targets = []
        convert_to_program = self.data_loader.gold_type_is_qdmr()

        # Sets the model to evaluation mode.
        self.valid_metrics.reset()
        with tqdm(total=len(self.data_loader)) as progbar:
            for batch_idx, (_, data, target, lexicon_str) in enumerate(self.data_loader):
                data, mask_data = batch_to_tensor(self.vocab, data, self.question_pad_length, self.device)
                target, mask_target = batch_to_tensor(self.vocab, target, self.qdmr_pad_length, self.device)
                lexicon_ids, mask_lexicon = tokenize_lexicon_str(self.vocab, lexicon_str, self.qdmr_pad_length, self.device)
                start = time.time()
                # Run the model on the batch and calculate the loss
                output, mask_output = self.model(data, target, lexicon_ids, evaluation_mode=True)
                loss = self.criterion(output, mask_output, target, mask_target)
                output = torch.transpose(output, 1, 2)
                pred = torch.argmax(output, dim=1)

                start = time.time()
                # Convert the predictions/ targets/questions from tensor of token_ids to list of strings.
                # TODO do we need to convert here or can we use the originals? (for data and target)
                data_str = batch_to_str(self.vocab, data, mask_data, convert_to_program=False)
                target_str = batch_to_str(self.vocab, target, mask_target, convert_to_program=convert_to_program)
                pred_str = pred_batch_to_str(self.vocab, pred, convert_to_program=convert_to_program)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_str, target_str, data_str))

                # Print example for predictions.
                if batch_idx == batch_index1:
                    questions.append(data_str[example_index1])
                    decompositions.append(pred_str[example_index1])
                    targets.append(target_str[example_index1])

                if batch_idx == batch_index2:
                    questions.append(data_str[example_index2])
                    decompositions.append(pred_str[example_index2])
                    targets.append(target_str[example_index2])

                # Update the progress bar.
                progbar.update(1)
                progbar.set_postfix(LOSS=loss.item(),
                                    batch_size=self.data_loader.init_kwargs['batch_size'],
                                    samples=self.data_loader.n_samples)

        # Print example predictions.
        for question, decomposition, target in zip(questions, decompositions, targets):
            print('\ndecomposition example:')
            print('question:\t\t', question)
            print('decomposition:\t', decomposition)
            print('target:\t\t\t', target)
            print()

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


    def _predict_without_target(self):
        """
        get model predictions for testing.
        Used without targets

        :return: A log that contains information about predictions
        """
        qid_col = []
        pred_col = []
        question_col = []

        convert_to_program = self.data_loader.gold_type_is_qdmr()

        # Sets the model to evaluation mode.
        self.valid_metrics.reset()
        with tqdm(total=len(self.data_loader)) as progbar:
            for batch_idx, (question_ids, data, target, lexicon_str) in enumerate(self.data_loader):
                data, mask_data = batch_to_tensor(self.vocab, data, self.question_pad_length, self.device)
                target, mask_target = batch_to_tensor(self.vocab, target, self.qdmr_pad_length, self.device)
                lexicon_ids, mask_lexicon = tokenize_lexicon_str(self.vocab, lexicon_str, self.qdmr_pad_length, self.device)
                start = time.time()
                # Run the model on the batch and calculate the loss
                output, mask_output = self.model(data, target, lexicon_ids, evaluation_mode=True)
                loss = self.criterion(output, mask_output, target, mask_target)
                output = torch.transpose(output, 1, 2)
                pred = torch.argmax(output, dim=1)
                start = time.time()
                # Convert the predictions/ targets/questions from tensor of token_ids to list of strings.
                # TODO do we need to convert here or can we use the originals? (for data and target)

                data_str = batch_to_str(self.vocab, data, mask_data, convert_to_program=False)
                target_str = batch_to_str(self.vocab, target, mask_target, convert_to_program=convert_to_program)
                pred_str = pred_batch_to_str(self.vocab, pred, convert_to_program=convert_to_program)

                for i, question_id in enumerate(question_ids):
                    self.logger.info('{}:{}'.format(question_id, data_str[i]))
                qid_col.extend(question_ids)
                pred_col.extend(pred_str)
                question_col.extend(data_str)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_str, target_str, data_str))

                # Update the progress bar.
                progbar.update(1)
                progbar.set_postfix(LOSS=loss.item(),
                                    batch_size=self.data_loader.init_kwargs['batch_size'],
                                    samples=self.data_loader.n_samples)
        d = {'question_id': qid_col, 'question_text': question_col, 'decomposition': pred_col}
        programs_df = pd.DataFrame(data=d)
        programs_df.to_csv(self.predictions_file_name, index=False, encoding='utf-8')

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
