# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao
# --------------------------------------------------------

import os
import logging
import time
import shutil
import atexit

def clear_symlink_Function(symlink_path):
    print "exiting the program and clear the enviroment"
    os.remove(symlink_path)

def create_logger(log_dir,cfg_name):

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(log_dir, cfg_name+".log"), format=head)
    logger = logging.getLogger()
    formatter = logging.Formatter(head)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def create_env(root_output_path, cfg, image_set):

    #final params dir
    image_sets = [iset for iset in image_set.split('+')]
    cfg_name = os.path.basename(cfg).split('.')[0]
    # if cfg_name.find('stage') > -1:
    #     cfg_name = cfg_name[:-7]
    config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    if not os.path.exists(config_output_path):
        os.makedirs(config_output_path)
    final_output_path = os.path.join(config_output_path, '{}'.format('_'.join(image_sets)))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    mtime = time.strftime('%Y-%m-%d-%H-%M')
    #experiments dir
    experiments_name = cfg_name +'_{}'.format(mtime)
    experiments_path = os.path.join(final_output_path, experiments_name)
    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)

    #assign the logger file and the tensorboard file
    shutil.copy2(cfg, os.path.join(experiments_path, cfg_name+'.yaml'))
    logger = create_logger(experiments_path,cfg_name)
    tensorboard_path = os.path.join(experiments_path,"tensorboard")

    #assign the log dir to the soft link to the experiments log
    experiments_log_dir = "./experiments_log"
    experiments_log_path = os.path.join(experiments_log_dir, cfg_name + '_{}'.format(mtime))
    if not os.path.exists(experiments_log_dir):
        os.makedirs(experiments_log_dir)
    true_experiments_log_path = os.path.join("..", experiments_path)
    os.symlink(true_experiments_log_path, experiments_log_path)

    # when exit clear the symlink path
    atexit.register(clear_symlink_Function, experiments_log_path)

    return logger, final_output_path, experiments_path ,tensorboard_path