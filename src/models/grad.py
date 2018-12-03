"""calculate grad
"""

import torch

def grad(target_activation, grad_val):
    """grad for each image
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
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear')
    cam = cam.view(224, 224)
    return cam
