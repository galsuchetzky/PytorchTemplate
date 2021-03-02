import os

configurations = [
    'configs/config_rnn_qdmr.json',
    'configs/config_rnn_qdmr_dynamic.json',
    'configs/config_rnn_qdmr_attention.json',
    'configs/config_rnn_qdmr_dynamic_attention.json',
    'configs/config_rnn_program.json',
    'configs/config_rnn_program_dynamic.json',
    'configs/config_rnn_program_attention.json',
    'configs/config_rnn_program_dynamic_attention.json'
]

command = 'python train.py -c '
debug = ' --debug True'

for configuration in configurations:
    experiment = command + configuration + debug
    print('running experiment: ', configuration)
    os.system(experiment)
