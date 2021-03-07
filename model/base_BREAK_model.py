import neuralcoref
import spacy

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
        :param verbose:
        :return: The QDMR.
        """
        raise NotImplementedError

    def load_decompositions_from_file(self, predictions_file):
        """
        Loades decompositions from a file.
        :param predictions_file: CSV file containing decompositions.
        """
        raise NotImplementedError

    def predict(self, questions, print_non_decomposed, verbose, extra_args=None):
        """
        Predicts the QDMRs for the given question.
        :param questions: The questions to decompose.
        :param print_non_decomposed: Should print the questions that were not decomposed?
        :param verbose:
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