"""FineTune"""

import torch.nn as nn
import torch.nn.functional as F

class FineTuneModel(nn.Module):
    """Wrap pretrin model to do finetuning and
       Make features function and classifier
    """

    def __init__(self, originalModel, args, logger):
        """init entire train model"""

        super(FineTuneModel, self).__init__()
        self.args = args
        self.logger = logger

        feature_modules = list(originalModel.children())[:-2]
        feature_modules = nn.Sequential(*feature_modules)
        self.features = feature_modules
        self.logger.debug('features: {}'.format(self.features))

        self.classifier = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, 7))
        self.avgpool = originalModel.avgpool
        self.gradients = []
        self.target_layers = ["7"]

    def get_gradients(self):
        """get_gradients"""
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

        self.logger.debug('inputs {}'.format(inputs.size()))
        outputs = inputs
        for name, module in self.features._modules.items():
            self.logger.debug('module {}'.format(module))
            outputs = module(outputs)
            self.logger.debug('outputs {}'.format(outputs.size()))
            if name in self.target_layers:
                outputs.register_hook(save_gradient)
                features += [outputs]

        outputs = self.avgpool(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        y_pred = self.classifier(outputs)

        return features, y_pred


    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = f.view(f.size(0), -1)
        self.logger.debug(f.size())
        y = self.classifier(F.relu(f))
        return y
