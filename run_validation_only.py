import os
from time import time
from os import listdir
from os.path import isfile, join
from subprocess import check_output

"""
usage: add in the config in the saved the line '"validate_only": true,' in the trainer part.
than add the metric (normalized exact match) to the eval metrics.
now running 
"""

paths = [
    'saved/models/SimpleRNN_Encoder_Decoder/minimized_domain_16',
    'saved/models/SimpleRNN_Encoder_Decoder/minimized_domain_256'
]

for path in paths:
    checkpoints = [checkpoint for checkpoint in listdir(path) if isfile(join(path, checkpoint)) and
                   checkpoint.startswith('checkpoint')]
    checkpoints = sorted(checkpoints,
                         key=lambda checkpoint: int(checkpoint.split('.')[0].replace('checkpoint-epoch','')))

    outputs = []
    for checkpoint in checkpoints:
        command = 'python train.py --resume ' + path + '/' + checkpoint

        print('evaluating: ', path.split('/')[-1], checkpoint)
        print()
        out = check_output(command)
        print(out.decode('utf-8').split('\r\n')[-2])
        outputs.append(out.decode('utf-8').split('\r\n')[-2])

    with open(path + '/validate_only.txt', 'w') as f:
        f.write('\n'.join(outputs))
    print()

    #todo need logic to keep the best over all checkpoints and save result to file
