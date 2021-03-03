import os
from time import time

configurations = [
    'configs/config_rnn_untied_qdmr.json',
    'configs/config_rnn_untied_program.json',

    'configs/config_rnn_tied_qdmr.json',
    'configs/config_rnn_tied_program.json',

    'configs/config_rnn_tied_qdmr_dynamic.json',
    'configs/config_rnn_tied_program_dynamic.json',

    'configs/config_rnn_tied_qdmr_dynamic_attention.json',
    'configs/config_rnn_tied_program_dynamic_attention.json'
]

command = 'python train.py -c '
debug = ' --debug True'

start_time = time()

for configuration in configurations:
    experiment = command + configuration # + debug
    print('running experiment: ', configuration)
    os.system(experiment)

end_time = time()

print("total experimenting time: ", end_time - start_time)