from abc import ABC, abstractmethod

import torch

from .encoder.builder import build_vision_tower
from .projector.builder import build_mm_projector


class VLMMetaModel:
    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_mm_projector(config)
            if getattr(config, "cls_loss_weight", 0.0) > 0:
                self.cls_head = torch.nn.Linear(config.hidden_size, config.cls_num_classes)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args):
        config = self.config
        config.input_size = list(model_args.input_size)
        config.dim = model_args.dim
        config.vision_tower = model_args.vision_tower
        config.vision_select_layer = model_args.vision_select_layer
        config.vision_select_feature = model_args.vision_select_feature
        config.mm_projector_type = model_args.mm_projector_type
        config.mm_mlp_depth = model_args.mm_mlp_depth
        config.mm_projector_norm = model_args.mm_projector_norm
        config.proj_out_num = model_args.proj_out_num
        config.ours_model_name = model_args.ours_model_name
        config.ours_feature_mode = model_args.ours_feature_mode
        config.ours_in_channels = model_args.ours_in_channels
        config.care_checkpoint_path = model_args.care_checkpoint_path
        config.care_text_assets_path = model_args.care_text_assets_path
        config.cls_loss_weight = 0.0
        config.cls_num_classes = 8
        # Kept for source-checkpoint field compatibility. New training uses care.pth.
        config.ours_resume_path = model_args.care_checkpoint_path

        if self.get_vision_tower() is not None:
            raise RuntimeError("Vision modules were already initialized")
        self.vision_tower = build_vision_tower(config)
        config.mm_hidden_size = self.vision_tower.hidden_size
        self.mm_projector = build_mm_projector(config)

        out_norm = getattr(self.mm_projector, "out_norm", None)
        if out_norm is not None:
            target = self.embed_tokens.weight.detach().float().std().item()
            torch.nn.init.constant_(out_norm.weight, target)


class VLMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        self._cached_projector_features = image_features
        return image_features

    def compute_cls_loss(self, cls_labels):
        feats = getattr(self, "_cached_projector_features", None)
        cls_head = getattr(self.get_model(), "cls_head", None)
        if feats is None or cls_head is None or cls_labels is None:
            return None
        pooled = feats.mean(dim=1)
        logits = cls_head(pooled.to(cls_head.weight.dtype))
        pos_weight = getattr(self.config, "cls_pos_weight", None)
        if pos_weight is not None:
            pos_weight = torch.as_tensor(
                pos_weight, device=logits.device, dtype=logits.dtype
            )
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            cls_labels.to(logits.dtype).to(logits.device),
            pos_weight=pos_weight,
        )

    def prepare_inputs_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        image_features = self.encode_images(images)
        inputs_embeds = self.get_model().embed_tokens(input_ids)
        inputs_embeds = torch.cat(
            (
                inputs_embeds[:, :1, :],
                image_features,
                inputs_embeds[:, (image_features.shape[1] + 1) :, :],
            ),
            dim=1,
        )
        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        )

    def initialize_vision_tokenizer(self, tokenizer, num_new_tokens):
        old_vocab_size = self.get_input_embeddings().weight.shape[0]
        self.resize_token_embeddings(len(tokenizer))
        new_vocab_size = self.get_input_embeddings().weight.shape[0]
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data
            input_embeddings[-num_new_tokens:] = input_embeddings[
                :-num_new_tokens
            ].mean(
                dim=0, keepdim=True
            )
            output_embeddings[-num_new_tokens:] = output_embeddings[
                :-num_new_tokens
            ].mean(
                dim=0, keepdim=True
            )
        return old_vocab_size, new_vocab_size
