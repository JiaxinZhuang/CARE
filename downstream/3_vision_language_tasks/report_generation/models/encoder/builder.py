import torch.nn as nn


def build_vision_tower(config, **kwargs):
    return VisionTower(config)


class VisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.vision_tower != "ours":
            raise ValueError(
                "The report-generation release only supports the CARE vision tower"
            )

        from .ours import create_ours

        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature
        self.hidden_size = config.dim
        self.vision_tower = create_ours(
            input_size=config.input_size,
            hidden_dim=config.dim,
            feature_mode=getattr(config, "ours_feature_mode", "tokens_cls"),
            in_channels=getattr(config, "ours_in_channels", 1),
            care_checkpoint_path=getattr(
                config,
                "care_checkpoint_path",
                getattr(config, "ours_resume_path", None),
            ),
            care_text_assets_path=getattr(config, "care_text_assets_path", None),
        )

    def forward(self, images):
        hidden_states = self.vision_tower(images)
        if self.select_layer == 0:
            image_features = hidden_states
        elif self.select_layer < 0:
            image_features = hidden_states[self.select_layer :]
        else:
            raise ValueError(f"Unexpected select layer: {self.select_layer}")

        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
