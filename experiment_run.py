import os
from time import time

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
command = 'python train.py -c '
debug = ' --debug True'

start_time = time()

for configuration in best_configs:
    experiment = command + configuration #+ debug

    print('running experiment: ', configuration)
    os.system(experiment)

end_time = time()

print("total experimenting time: ", end_time - start_time)
