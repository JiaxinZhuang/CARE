"""
Written By Lincolnzjx
"""
import os
import argparse
import logging
#import pretrainedmodels
import torchvision
from models.FineTuneModel import FineTuneModel
from models.models import ConvNetTrain


def add_parser():
    """add parser to get hyper parameters
    """
    parser = argparse.ArgumentParser(description="Hyperparameter")

    parser.add_argument('--cuda', default='0',
                        help="cuda visible device")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help="train from a setted epoch")
    parser.add_argument('--n_epochs', type=int, default=401,
                        help="all epochs")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="lr")
    parser.add_argument('--iterNo', default=1,
                        help="iter No")
    parser.add_argument('--data_dir', default="../data/",
                        help="data dir")
    parser.add_argument('--batch_size', type=int, default=96,
                        help="batch size")
    parser.add_argument('--logfile', default='result',
                        help="save log dir")
    parser.add_argument('--train_dir', default='../train_dir',
                        help="save train model dir")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="alpha")
    parser.add_argument('--inner_threshold', type=float, default=0.5,
                        help="inner_threshold for mask")
    parser.add_argument('--stage', type=int, default=1, \
                        help='stage one or stage two')
    parser.add_argument('--stage_one', type=str, default='pretrained', \
                        help='stage one load model from pretrain or saved model')
    parser.add_argument('--stage_two', type=str, default='train_dir_006_11/40', \
                        help='stage two load model from saved model')
    parser.add_argument('--bash_name', type=str, default='run_', \
                        help='run bash name')

    return parser


def main(args, logger):
    """main module where select models and loads
    """

    net = torchvision.models.resnet50(pretrained=True)
    #net = pretrainedmodels.resnet50(pretrained=True)
    finetune_model = FineTuneModel(net, args, logger)

    convnet_train = ConvNetTrain(finetune_model, args, logger=logger)
    convnet_train.iterate_convnetwork()


if __name__ == '__main__':
    ARGS = add_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.cuda

    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger('logger')
    main(ARGS, LOGGER)
