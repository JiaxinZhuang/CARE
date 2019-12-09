"""Trainer.

    Jiaxin Zhuang, lincolnz9511@gmail.com
"""
# -*- coding: utf-8 -*-

import os
import sys
# import itertools
# import shutil


from PIL import Image
# import sklearn
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
# import matplotlib.pyplot as plt
# import numpy as np


import config  # TODO
import dataset  # TODO
import model  # TODO
from loss import GradLoss, grad
from utils.function import onehot


def main():
    """Main.
    """
    configs = config.Config()
    configs_dict = configs.get_config()
    exp = configs_dict["experiment_index"]
    cuda_id = configs_dict["cuda"]
    num_workers = configs_dict["num_workers"]
    seed = configs_dict["seed"]
    n_epochs = configs_dict["n_epochs"]
    log_dir = configs_dict["log_dir"]
    model_dir = configs_dict["model_dir"]
    batch_size = configs_dict["batch_size"]
    learning_rate = configs_dict["learning_rate"]
    eval_frequency = configs_dict["eval_frequency"]
    resume = configs_dict["resume"]
    optimizer = configs_dict["optimizer"]
    input_size = configs_dict["input_size"]
    re_size = configs_dict["re_size"]
    backbone = configs_dict["backbone"]
    dataset_name = configs_dict["dataset"]
    stage = configs_dict["stage"]
    alpha = configs_dict["alpha"]
    inner_threshold = configs_dict["inner_threshold"]

    # init environment and log
    init_environment(seed=seed, cuda_id=cuda_id)
    _print = init_logging(log_dir, exp).info
    configs.print_config(_print)
    tf_log = os.path.join(log_dir, exp)
    writer = SummaryWriter(log_dir=tf_log)
    try:
        os.mkdir(os.path.join(model_dir, exp))
    except FileExistsError:
        _print("{} has been created.".format(os.path.join(model_dir, exp)))

    # dataset
    _print(">> Dataset:{} - Input size: {}".format(dataset_name, input_size))
    if dataset_name == "skin7":
        num_classes = 7
        mean = [0.7626, 0.5453, 0.5714]
        std = [0.1404, 0.1519, 0.1685]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        val_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        trainset = dataset.Skin7(root="./data/", is_train=True,
                                 transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)
        valset = dataset.Skin7(root="./data/", is_train=False,
                               transform=val_transform)
        valloader = torch.utils.data.DataLoader(valset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=num_workers)
    else:
        _print("Need dataset.")
        sys.exit(-1)

    net = model.network(backbone=backbone, num_classes=num_classes,
                        _print=_print)
    net = net.cuda()
    net.print_model()

    # Optimizer & loss function
    if optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(),
                              lr=learning_rate,
                              momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.1, patience=10,
                        verbose=True, threshold=1e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = None

    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_grad = GradLoss(inner_threshold)

    # TODO
    # load dict for model
    start_epoch = 0

    # Train, stage 1 or stage 2 need to decide by parameter
    train_cam_losses = 0
    train_ce_losses = 0
    for epoch in range(start_epoch, n_epochs):
        net.train()
        for batch_idx, (inputs, targets, has_masks, inner_batches,
                        outer_batches) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            features, outputs = net.extractor(inputs)
            ce_losses = criterion_ce(outputs, targets)
            # _, pred = torch.max(outputs.data, 1)
            if stage == 2:
                targets_onehot = onehot(targets, num_classes)
                targets_value = targets_onehot * outputs
                sum_output = torch.sum(targets_value)
                sum_output.backward(retain_graph=True)
                grad_vals = net.get_gradients()[-1]
                # Loss:
                index = 0
                cam_losses = torch.tensor(0.0).cuda()
                for has_mask, feature, grad_val, inner, outer in\
                    zip(has_masks, features[0], grad_vals, inner_batches,
                        outer_batches):
                    if has_mask:
                        cam = grad(feature, grad_val).cuda()
                        inner, outer = inner.cuda(), outer.cuda()
                        cam_loss = criterion_grad(cam, inner, outer).cuda()
                        cam_losses += cam_loss
                    index += 1

                total_losses = alpha * cam_losses + (1-alpha) * ce_losses

                train_cam_losses += alpha.item() * cam_losses.item()
                train_ce_losses += (1-alpha.item()) * ce_losses.item()
            else:
                train_cam_losses = 0.0
                train_ce_losses += ce_losses.item()
                total_losses = ce_losses
            total_losses.backward()
            optimizer.step()

        writer.add_scalar('train/ce_loss', train_ce_losses, epoch)
        writer.add_scalar('train/cam_loss', train_cam_losses, epoch)
        writer.add_scalar('train/total_loss', train_cam_losses+train_ce_losses,
                          epoch)

        # TODO: evaluation.
        # TODO: save the best model.
