import os
from time import time
from os import listdir
from os.path import isfile, join

configurations = [
    'configs/config_rnn_untied_qdmr.json',  # gru_untied_qdmr
    'configs/config_rnn_untied_program.json',  # gru_untied_program

    'configs/config_rnn_tied_qdmr.json',  # gru_tied_qdmr
    'configs/config_rnn_tied_program.json',  # gru_tied_program

    'configs/config_rnn_tied_qdmr_dynamic.json',  # gru_tied_qdmr_dynamic
    'configs/config_rnn_tied_program_dynamic.json',  # gru_tied_program_dynamic

    'configs/config_rnn_tied_qdmr_dynamic_attention.json',  # gru_tied_qdmr_dynamic_attention
    'configs/config_rnn_tied_program_dynamic_attention.json'  # gru_tied_program_dynamic_attention
]

test_configs = [
    'configs/config_rnn_tied_program_attention.json'
]

test_configurations = [
    'configs_Itay/config_rnn_tied_program.json'
]

best_configs = [
    'configs/config_rnn_tied_program_dynamic_attention_dropout.json',
    'configs/config_rnn_tied_qdmr_dynamic_attention_dropout.json'
]

capacity_configs_manual16 = [
    'configs_capacity/config_qdmr_vanilla_16.json',
    'configs_capacity/config_qdmr_length_16.json',
    'configs_capacity/config_qdmr_domain_16.json',
    'configs_capacity/config_program_vanilla_16.json',
    'configs_capacity/config_program_length_16.json',
    'configs_capacity/config_program_domain_16.json',
    'configs_capacity/config_minimized_vanilla_16.json',
    'configs_capacity/config_minimized_length_16.json',
    'configs_capacity/config_minimized_domain_16.json'
]

capacity_configs_manual64 = [
    'configs_capacity/config_qdmr_vanilla_64.json',
    'configs_capacity/config_qdmr_length_64.json',
    'configs_capacity/config_qdmr_domain_64.json',
    'configs_capacity/config_program_vanilla_64.json',
    'configs_capacity/config_program_length_64.json',
    'configs_capacity/config_program_domain_64.json',
    'configs_capacity/config_minimized_vanilla_64.json',
    'configs_capacity/config_minimized_length_64.json',
    'configs_capacity/config_minimized_domain_64.json'
]

capacity_configs_manual256 = [
    'configs_capacity/config_qdmr_vanilla_256.json',
    'configs_capacity/config_qdmr_length_256.json',
    'configs_capacity/config_qdmr_domain_256.json',
    'configs_capacity/config_program_vanilla_256.json',
    'configs_capacity/config_program_length_256.json',
    'configs_capacity/config_program_domain_256.json',
    'configs_capacity/config_minimized_vanilla_256.json',
    'configs_capacity/config_minimized_length_256.json',
    'configs_capacity/config_minimized_domain_256.json'
]

capacity_configs_manual1024 = [
    'configs_capacity/config_qdmr_vanilla_1024.json',
    'configs_capacity/config_qdmr_length_1024.json',
    'configs_capacity/config_qdmr_domain_1024.json',
    'configs_capacity/config_program_vanilla_1024.json',
    'configs_capacity/config_program_length_1024.json',
    'configs_capacity/config_program_domain_1024.json',
    'configs_capacity/config_minimized_vanilla_1024.json',
    'configs_capacity/config_minimized_length_1024.json',
    'configs_capacity/config_minimized_domain_1024.json'
]

capacity_configs_path = 'configs_capacity/'
capacity_configs = [conf for conf in listdir(capacity_configs_path) if isfile(join(capacity_configs_path, conf))]


command = 'python train.py -c '
debug = ' --debug True'

start_time = time()

for configuration in capacity_configs_manual16:
    experiment = command + configuration #+ debug

    print('running experiment: ', configuration)
    os.system(experiment)

end_time = time()

print("total experimenting time: ", end_time - start_time)
