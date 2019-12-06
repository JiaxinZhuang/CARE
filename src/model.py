"""Model for CARE.
"""

import torch.nn as nn
import torch.nn.functional as F


class GradModel(nn.Module):
    """In order to save gradients during training, modifying model here.
       Wrap pretrained-model, make features function and classifier
     """
    def __init__(self, original_model, num_classes=7):
        super(GradModel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[: -2])
        self.avgpool = original_model.avgpool
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, num_classes)
        )
        self.gradients = []
        self.target_layers = ["7"]  # for resnet50

    def get_gradients(self, inputs):
        """Get gradients.
        """
        return self.gradients

    def extractor(self, inputs):
        """extract features and outputs
        Args:
            inputs: images [batch_size, channel, height, width]
        Returns:
            features:
            outputs:
        """
        def save_gradient(grad):
            """save_gradient"""
            self.gradients.append(grad)

        features = []
        self.gradients = []

        # print("Input size: {}".format(inputs.size())
        outputs = inputs
        for name, module in self.features._modules.items():
            outputs = module(outputs)
            if name in self.target_layers:
                outputs.register_hook(save_gradient)
                features += [outputs]

        outputs = self.avgpool(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        y_pred = self.classifier(outputs)
        return features, y_pred

    def forward(self, inputs):
        out = self.features(inputs)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(F.relu(out))
        return out
