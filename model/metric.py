import torch


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
        correct = torch.sum(torch.all(torch.eq(pred, target),dim=1))

    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
