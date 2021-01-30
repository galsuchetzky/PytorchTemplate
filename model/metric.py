import torch
import numpy as np
from tester.BREAK_evaluation.sari_hook import get_sari
from tester.BREAK_evaluation.sequence_matcher import SequenceMatchScorer
from tester.BREAK_evaluation.graph_matcher import GraphMatchScorer
from tester.BREAK_evaluation.decomposition import Decomposition


# TODO bring all the metrics from the BREAK repo.
def accuracy_MNIST(output, target):
    """
    Tests the accuracy of the prediction to the target.
    :param output: The prediction of the model.
    :param target: The gold target of the prediction.
    :return: The percent of correct predictions from the whole.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()

    return correct / len(target)


def accuracy_qdmr(pred, target):
    """
    Tests the EM accuracy of a QDMR batch prediction.
    :param pred: A batch of predicted QDMRs. dim: (batch_size, seq_len)
    :param target: The gold QDMRs. dim: (batch_size, seq_len)
    :return: The percent of correct predictions from the whole batch.
    """
    # TODO make it work correctly on a batch and handle the predictions correctly.
    #  Maybe take from the break original code.
    with torch.no_grad():
        assert pred.shape == target.shape
        correct = torch.sum(torch.all(torch.eq(pred, target), dim=1)).item()

    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def exact_match(decompositions_str: [str], golds_str: [str], *args):
    return sum([d.lower() == g.lower() for d, g in zip(decompositions_str, golds_str)]) / len(decompositions_str)


def sari_score(decompositions_str: [str], golds_str: [str], questions: [str]):
    sources = [q.split(" ") for q in questions]
    predictions = [d.split(" ") for d in decompositions_str]
    targets = [[g.split(" ")] for g in golds_str]
    sari, keep, add, deletion = get_sari(sources, predictions, targets)
    return np.average(sari)


def match_ratio(decompositions_str: [str], golds_str: [str], *args):
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    scores = sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                              processing="base")
    return np.average(scores)


def structural_match_ratio(decompositions_str: [str], golds_str: [str], *args):
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    scores = sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                              processing="structural")
    return np.average(scores)


def ged_score(decompositions_str: [str], golds_str: [str], *args):
    decompositions = [Decomposition.from_str(decomp) for decomp in decompositions_str]
    golds = [Decomposition.from_str(g) for g in golds_str]

    graph_scorer = GraphMatchScorer()
    decomposition_graphs = [d.to_graph() for d in decompositions]
    gold_graphs = [g.to_graph() for g in golds]

    ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs)

    return np.average(ged_scores)
