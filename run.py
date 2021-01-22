import torch
import model.loss as module_loss
import model.metric as module_metric
import trainer.trainer as trainer_module
import tester.tester as tester_module

from utils import prepare_device


def run(config, train=True):
    """
    :param config:
    :param train:
    :return:
    """
    logger = config.get_logger('train' if train else 'test')

    # setup data_loader instances
    if not train:
        config['data_loader']['args']['training'] = False
    data_loader = config.init_obj('data_loader')

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    logger.info(f"Working on device: {device}")

    # build model architecture, then print to console
    config['arch']['args']['device'] = device  # Add the device to the config.
    model = config.init_obj('arch')
    logger.info(model)

    # Move model to device.
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])()
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    if train:
        valid_data_loader = data_loader.split_validation()

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        cfg_trainer = config['trainer']
        trainer_class = getattr(trainer_module, cfg_trainer['type'])

        trainer = trainer_class(model, criterion, metrics, optimizer,
                                config=config,
                                device=device,
                                data_loader=data_loader,
                                valid_data_loader=valid_data_loader,
                                lr_scheduler=lr_scheduler)

        trainer.train()
    else:
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Checkpoint loaded.")

        cfg_tester = config['tester']
        tester_class = getattr(tester_module, cfg_tester['type'])

        tester = tester_class(model, criterion, metrics,
                              config=config,
                              device=device,
                              data_loader=data_loader)

        tester.test()


