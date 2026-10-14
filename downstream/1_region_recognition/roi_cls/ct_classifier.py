"""Linear ROI classifier on top of the shared CARE visual encoder."""

import torch.nn as nn

if __package__:
    from .models.ours import DEFAULT_CARE_CHECKPOINT, create_ours
else:
    from models.ours import DEFAULT_CARE_CHECKPOINT, create_ours


class CT3DClassifier(nn.Module):
    """End-to-end CT classifier with the original linear-head contract."""

    def __init__(
        self,
        encoder_name="ours",
        num_classes=None,
        pretrained_path=DEFAULT_CARE_CHECKPOINT,
        freeze_encoder=True,
        **encoder_kwargs,
    ):
        super().__init__()
        if encoder_name.lower() != "ours":
            raise ValueError("Only the CARE encoder is supported.")
        if num_classes is None or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")

        self.encoder_name = encoder_name.lower()
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.encoder = create_ours(
            pretrained_path=pretrained_path,
            **encoder_kwargs,
        )
        self.feature_dim = self.encoder.output_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
        self.freeze_encoder = False

    def forward(self, image):
        return self.classifier(self.encoder(image))

    def get_features(self, image):
        return self.encoder(image)
