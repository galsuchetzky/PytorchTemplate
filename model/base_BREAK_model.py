import numpy as np
import neuralcoref
import pandas as pd
import spacy
import torch.nn as nn

from tester.BREAK_evaluation.decomposition import Decomposition
from .base_model import BaseModel


class BaseBREAKModel(BaseModel):
    """
    Base model for Break models.
    """

    def __init__(self, device):
        """
        Initiates the base model.
        :param device: The device to move the model to.
        """
        super().__init__(device)

        # Load a spacy model for tokenization.
        self.parser = spacy.load('en_core_web_sm')

        # TODO check what is the usage of this.
        coref = neuralcoref.NeuralCoref(self.parser.vocab)
        self.parser.add_pipe(coref, name='neuralcoref')

    def forward(self, *inputs):
        """
        Forward pass logic.
        :param inputs: the inputs to the forward pass.
        :return: Model output.
        """
        raise NotImplementedError

    def _parse(self, question):
        """Run a spaCy model for dependency parsing, POS tagging, etc."""
        return self.parser(question)

    def _decompose(self, question, verbose):
        """
        Decomposes a question into it's corresponding QDMR.
        :param question: The question to decompose.
        :param verbose: TODO what is this used for?
        :return: The QDMR.
        """
        raise NotImplementedError

    def load_decompositions_from_file(self, predictions_file):
        """
        Loades decompositions from a file.
        :param predictions_file: CSV file containing decompositions.
        :return: TODO list of decompositions?
        """
        raise NotImplementedError

    def predict(self, questions, print_non_decomposed, verbose, extra_args=None):
        """
        Predicts the QDMRs for the given question.
        :param questions: The questions to decompose.
        :param print_non_decomposed: Should print the questions that were not decomposed?
        :param verbose: TODO verbosity to use?
        :param extra_args: extra args.
        :return: A list of Decomposition objects with the created decompositions.
        """
        # List for the generated QDMRs.
        decompositions = []

        # Decompose the questions, count the decomposed and the non-decomposed.
        num_decomposed, num_not_decomposed = 0, 0
        for question in questions:
            # Decompose the question.
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

    # @staticmethod
    # def print_score_stats(evaluation_dict):
    #     print("\noverall scores:")
    #
    #     for key in evaluation_dict:
    #         # ignore keys that do not store scores
    #         if key in ["question", "gold", "prediction"]:
    #             continue
    #         score_name, scores = key, evaluation_dict[key]
    #
    #         # ignore examples without a score
    #         if None in scores:
    #             scores_ = [score for score in scores if score is not None]
    #         else:
    #             scores_ = scores
    #
    #         mean_score, max_score, min_score = np.mean(scores_), np.max(scores_), np.min(scores_)
    #         print("{} score:\tmean {:.3f}\tmax {:.3f}\tmin {:.3f}".format(
    #             score_name, mean_score, max_score, min_score))

    # TODO use this?
    # @staticmethod
    # def write_evaluation_output(output_path_base, num_examples, **kwargs):
    #     # write evaluation summary
    #     with open(output_path_base + '_summary.tsv', 'w') as fd:
    #         fd.write('\t'.join([key for key in sorted(kwargs.keys())]) + '\n')
    #         for i in range(num_examples):
    #             fd.write('\t'.join([str(kwargs[key][i]) for key in sorted(kwargs.keys())]) + '\n')
    #
    #     # write evaluation scores per example
    #     df = pd.DataFrame.from_dict(kwargs, orient="columns")
    #     df.to_csv(output_path_base + '_full.tsv', sep='\t', index=False)
