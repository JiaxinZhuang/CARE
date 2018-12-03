"""GradLoss
"""

import sys
import torch
from torch import nn

class GradLoss(nn.Module):
    """GradLoss
    """

    def __init__(self, inner_threshold, logger):
        super(GradLoss, self).__init__()
        self.inner_threshold = torch.tensor(inner_threshold).cuda()
        self.logger = logger

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
            self.logger.error('grad {}'.format(grad.size()))
            self.logger.error('grad {}'.format(grad))
            self.logger.error('inner {}'.format(inner.size()))
            self.logger.error('inner {}'.format(inner))
            self.logger.error('inner_loss: {}'.format(inner_loss))
            sys.exit(-1)
        if inner_loss > self.inner_threshold:
            #inner_loss = torch.Tensor(inner_threshold).squeeze()
            inner_loss = self.inner_threshold.squeeze()
        outer_loss = (grad * outer).sum()
        grad_loss = -inner_loss + outer_loss
        return grad_loss

    #loss_sum = 0
    #for grad, inner, outer in zip(grad_batch, inner_batch, outer_batch):
    #    inner_loss = (grad * inner).sum()
    #    if inner_loss > inner_threshold:
    #        inner_loss = torch.Tensor(inner_threshold).squeeze()
    #    outer_loss = (grad * outer).sum()
    #    loss = -inner_loss + outer_loss
    #    loss_sum = loss_sum + loss
    #return loss_sum
