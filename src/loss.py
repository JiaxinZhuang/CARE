"""GradLoss
"""

import sys
import torch
from torch import nn


class GradLoss(nn.Module):
    """GradLoss
    """

    def __init__(self, inner_threshold):
        super(GradLoss, self).__init__()
        self.inner_threshold = torch.tensor(inner_threshold).cuda()

    def forward(self, grad_batch, inner_batch, outer_batch):
        return self.grad_loss(grad_batch, inner_batch, outer_batch)

    def grad_loss(self, grad, inner, outer):
        """accept inner, outer and grad , return a loss, height=width=224
        Args:
            inner_threshold: tensor, scalar
            grad: [height, width]
            inner: [height, width]
            outer: [height, width]
        Return:
            grad_loss: tensor, [batch_size]
        """

        inner_loss = (grad * inner).sum()
        if torch.isnan(inner_loss):
            print(grad.size())
            print(grad)
            print(inner.size())
            print(inner)
            print(inner_loss)
            sys.exit(-1)
        if inner_loss > self.inner_threshold:
            inner_loss = self.inner_threshold.squeeze()
        outer_loss = (grad * outer).sum()
        grad_loss = -inner_loss + outer_loss
        return grad_loss


def grad(target_activation, grad_val):
    """Grad for each image.
    Args:
        target_activation: [2048, 7, 7]
        output: [7]
        grad_val: [2048, 7, 7]
    Return:
        grad: [224,224]
    """

    # [2048, 7, 7]
    channels, height, width = grad_val.size()
    # [2048, 49]
    grad_val = grad_val.view(channels, -1)
    weights = torch.mean(grad_val, dim=1)
    # [2048]
    weights = weights.view(channels, 1, 1)
    cam = weights * target_activation

    # [2048]
    cam = torch.sum(cam, dim=0)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    cam = cam.view(1, 1, height, width)
    cam = torch.nn.functional.interpolate(cam, size=(224, 224),
                                          mode='bilinear')
    cam = cam.view(224, 224)
    return cam
