"""Function.

    Jiaxin Zhuang, lincolnz9511@gmail.com
"""


import os
import sys
import logging
import time
from functools import wraps

import numpy as np
import torch


def onehot(inputs, num_classes):
    """Onehot.
        convert labels tensor to onehot tensors.
    Args:
        inputs: [batch_size]
        num_classes: int, size of classes
    Returns:
        inputs_onehot: [batch_size, num_classes]
    """
    num_sample = inputs.size(0)
    inputs_onehot = torch.zeros(num_sample, num_classes)
    inputs_onehot.scatter_(1, inputs.unsqueeze(1), 1.0)
    return inputs_onehot


def init_environment(seed=0, cuda_id=0):
    """Init environment
    initialize environment including cuda, benchmark, random seed, saved model
    directory and saved logs directory
    """
    print(">< init_environment with seed: {}".format(seed))

    cuda_id = str(cuda_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id

    if seed != -1:
        print("> Use seed -{}".format(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        print("> Don't use seed")

    # log_dir="./saved/logs/", model_dir="./saved/models/", \
    #  remove and create logdir
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # os.mkdir(log_dir)

    #  remove and create modeldir
    # if os.path.exists(model_dir):
    #     shutil.rmtree(model_dir)
    # os.mkdir(model_dir)


def init_logging(output_dir, exp):
    """Init logging and return logging
        init_logging should used after init environment
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(message)s",
                        datefmt="%Y%m%d-%H:%M:%S",
                        filename=os.path.join(output_dir, str(exp) + ".log"),
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    return logging


def timethis(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        elapse = time.time() - start_time
        print(">> Functoin: {} costs {:.4f}s".format(func.__name__, elapse))
        sys.stdout.flush()
        return ret
    return wrapper


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    inputs = torch.arange(5)
    num_classes = 5
    print(onehot(inputs, num_classes))
