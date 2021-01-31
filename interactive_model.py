import argparse
import torch

from parse_config import ConfigParser
from utils.util import prepare_device
from data_loader.vocabs import batch_to_tensor, batch_to_str, pred_batch_to_str


def main(config):
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    print(f"Working on device: {device}")

    # build model architecture, then print to console
    config['arch']['args']['device'] = device  # Add the device to the config.
    model = config.init_obj('arch')
    print(model)

    assert config.resume is not None, 'resume model should be provided'
    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    batch_size = model.batch_size
    question_pad_length = config['data_loader']['question_pad_length']
    qdmr_pad_length = config['data_loader']['qdmr_pad_length']

    while True:
        question = input('question: ')
        # TODO change this to "predict" function
        questions = [[question]] + [['']] * (batch_size - 1)
        data, mask_data = batch_to_tensor(model.vocab, questions, question_pad_length, device)

        targets = [['']] * batch_size
        target, mask_target = batch_to_tensor(model.vocab, targets, qdmr_pad_length, device)

        output = model(data, target)

        decomposition = pred_batch_to_str(model.vocab, output)[0]

        print('decomposition:\t', decomposition)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-db', '--debug', default=False,
                      help='add this argument to use a small subset of the dataset, for debugging.')

    config = ConfigParser.from_args(args)
    main(config)
