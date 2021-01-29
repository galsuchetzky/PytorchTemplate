import torch
from tqdm import tqdm

from .base_tester import BaseTester
from utils.util import MetricTracker
from data_loader.vocabs import batch_to_tensor, batch_to_str, pred_batch_to_str
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
            for batch_idx, (_, data, target) in enumerate(self.data_loader):
                data, mask_data = batch_to_tensor(self.vocab, data, self.question_pad_length, self.device)
                target, mask_target = batch_to_tensor(self.vocab, target, self.qdmr_pad_length, self.device)

                # Run the model on the batch and calculate the loss
                output = self.model(data, target, evaluation_mode=True)
                output = torch.transpose(output, 1, 2)
                pred = torch.argmax(output, dim=1)
                loss = self.criterion(output, target)

                # Convert the predictions/ targets/questions from tensor of token_ids to list of strings.
                data_str = batch_to_str(self.vocab, data, mask_data)
                target_str = batch_to_str(self.vocab, target, mask_target)
                pred_str = pred_batch_to_str(self.vocab, pred)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_str, target_str, data_str))

                # Update the progress bar.
                progbar.update(1)
                progbar.set_postfix(LOSS=loss.item(),
                                    batch_size=self.data_loader.init_kwargs['batch_size'],
                                    samples=self.data_loader.n_samples)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    #TODO add predict function for submission. do not use target

    # def batch_evaluate(self, questions, decompositions, golds, metadata,
    #                    output_path_base, num_processes):
    #     decompositions_str = [d.to_string() for d in decompositions]
    #     golds_str = [g.to_string() for g in golds]
    #
    #     # calculating exact match scores
    #     exact_match = [d.lower() == g.lower() for d, g in zip(decompositions_str, golds_str)]
    #
    #     # evaluate using SARI
    #     sources = [q.split(" ") for q in questions]
    #     predictions = [d.split(" ") for d in decompositions_str]
    #     targets = [[g.split(" ")] for g in golds_str]
    #     sari, keep, add, deletion = get_sari(sources, predictions, targets)
    #
    #     # evaluate using sequence matcher
    #     sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    #     match_ratio = sequence_scorer.get_match_scores(decompositions_str, golds_str,
    #                                                    processing="base")
    #     structural_match_ratio = sequence_scorer.get_match_scores(decompositions_str, golds_str,
    #                                                               processing="structural")
    #
    #     # evaluate using graph distances
    #     graph_scorer = GraphMatchScorer()
    #     decomposition_graphs = [d.to_graph() for d in decompositions]
    #     gold_graphs = [g.to_graph() for g in golds]
    #
    #     ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs)
    #     structural_ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs,
    #                                                                         structure_only=True)
    #     ged_plus_scores = get_ged_plus_scores(decomposition_graphs, gold_graphs,
    #                                           exclude_thr=5, num_processes=num_processes)
    #
    #     evaluation_dict = {
    #         "question": questions,
    #         "gold": golds_str,
    #         "prediction": decompositions_str,
    #         "exact_match": exact_match,
    #         "match": match_ratio,
    #         "structural_match": structural_match_ratio,
    #         "sari": sari,
    #         "ged": ged_scores,
    #         "structural_ged": structural_ged_scores,
    #         "ged_plus": ged_plus_scores
    #     }
    #     num_examples = len(questions)
    #     self.print_first_example_scores(evaluation_dict, min(5, num_examples))
    #     self.print_score_stats(evaluation_dict)
    #     print("skipped {} examples when computing ged.".format(
    #         len([score for score in ged_scores if score is None])))
    #     print("skipped {} examples when computing structural ged.".format(
    #         len([score for score in structural_ged_scores if score is None])))
    #     print("skipped {} examples when computing ged plus.".format(
    #         len([score for score in ged_plus_scores if score is None])))
    #
    #     if output_path_base:
    #         self.write_evaluation_output(output_path_base, num_examples, **evaluation_dict)
    #
    #     if metadata is not None:
    #         metadata = metadata[metadata["question_text"].isin(evaluation_dict["question"])]
    #         metadata["dataset"] = metadata["question_id"].apply(lambda x: x.split("_")[0])
    #         metadata["num_steps"] = metadata["decomposition"].apply(lambda x: len(x.split(";")))
    #         score_keys = [key for key in evaluation_dict if key not in ["question", "gold", "prediction"]]
    #         for key in score_keys:
    #             metadata[key] = evaluation_dict[key]
    #
    #         for agg_field in ["dataset", "num_steps"]:
    #             df = metadata[[agg_field] + score_keys].groupby(agg_field).agg("mean")
    #             print(df.round(decimals=3))
    #
    #     return evaluation_dict
