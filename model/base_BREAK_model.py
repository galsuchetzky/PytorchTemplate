import numpy as np
# import neuralcoref
import pandas as pd
import spacy
import torch.nn as nn

from tester.BREAK_evaluation.decomposition import Decomposition
from tester.BREAK_evaluation.graph_matcher import GraphMatchScorer, get_ged_plus_scores
from tester.BREAK_evaluation.sari_hook import get_sari
from tester.BREAK_evaluation.sequence_matcher import SequenceMatchScorer
from .base_model import BaseModel


class BaseBREAKModel(BaseModel):
    def __init__(self, device):
        super().__init__(device)
        self.parser = spacy.load('en_core_web_sm')
        # coref = neuralcoref.NeuralCoref(self.parser.vocab)
        # self.parser.add_pipe(coref, name='neuralcoref')

    def forward(self, *inputs):
        """
        Forward pass logic
        :param inputs: the inputs to the forward pass.
        :return: Model output
        """
        raise NotImplementedError

    def _parse(self, question):
        """Run a spaCy model for dependency parsing, POS tagging, etc."""
        return self.parser(question)

    def _decompose(self, question, verbose):
        raise NotImplementedError

    def load_decompositions_from_file(self, predictions_file):
        raise NotImplementedError

    def predict(self, questions, print_non_decomposed, verbose, extra_args=None):
        decompositions = []
        num_decomposed, num_not_decomposed = 0, 0
        for question in questions:
            decomposed, trace = self._decompose(question, verbose)
            if len(decomposed) == 1:
                num_not_decomposed += 1
                if print_non_decomposed:
                    print("question: {}\ndecomposition: -\n".format(question))
            else:
                num_decomposed += 1
                print("question: {}\ndecomposition: {}\ntrace: {}\n".format(question, decomposed, trace))

            decompositions.append(decomposed)

        print("\n{} decomposed questions, {} not-decomposed questions.\n".format(num_decomposed, num_not_decomposed))
        return [Decomposition(d) for d in decompositions]

    @staticmethod
    def print_first_example_scores(evaluation_dict, num_examples):
        for i in range(num_examples):
            print("evaluating example #{}".format(i))
            print("\tsource (question): {}".format(evaluation_dict["question"][i]))
            print("\tprediction (decomposition): {}".format(evaluation_dict["prediction"][i]))
            print("\ttarget (gold): {}".format(evaluation_dict["gold"][i]))
            print("\texact match: {}".format(round(evaluation_dict["exact_match"][i], 3)))
            print("\tmatch score: {}".format(round(evaluation_dict["match"][i], 3)))
            print("\tstructural match score: {}".format(round(evaluation_dict["structural_match"][i], 3)))
            print("\tsari score: {}".format(round(evaluation_dict["sari"][i], 3)))
            print("\tGED score: {}".format(
                round(evaluation_dict["ged"][i], 3) if evaluation_dict["ged"][i] is not None
                else '-'))
            print("\tstructural GED score: {}".format(
                round(evaluation_dict["structural_ged"][i], 3) if evaluation_dict["structural_ged"][i] is not None
                else '-'))
            print("\tGED+ score: {}".format(
                round(evaluation_dict["ged_plus"][i], 3) if evaluation_dict["ged_plus"][i] is not None
                else '-'))

    @staticmethod
    def print_score_stats(evaluation_dict):
        print("\noverall scores:")

        for key in evaluation_dict:
            # ignore keys that do not store scores
            if key in ["question", "gold", "prediction"]:
                continue
            score_name, scores = key, evaluation_dict[key]

            # ignore examples without a score
            if None in scores:
                scores_ = [score for score in scores if score is not None]
            else:
                scores_ = scores

            mean_score, max_score, min_score = np.mean(scores_), np.max(scores_), np.min(scores_)
            print("{} score:\tmean {:.3f}\tmax {:.3f}\tmin {:.3f}".format(
                score_name, mean_score, max_score, min_score))

    @staticmethod
    def write_evaluation_output(output_path_base, num_examples, **kwargs):
        # write evaluation summary
        with open(output_path_base + '_summary.tsv', 'w') as fd:
            fd.write('\t'.join([key for key in sorted(kwargs.keys())]) + '\n')
            for i in range(num_examples):
                fd.write('\t'.join([str(kwargs[key][i]) for key in sorted(kwargs.keys())]) + '\n')

        # write evaluation scores per example
        df = pd.DataFrame.from_dict(kwargs, orient="columns")
        df.to_csv(output_path_base + '_full.tsv', sep='\t', index=False)


