from .mlp import MultiModalProjector


def build_mm_projector(config, delay_load=False, **kwargs):
    if config.mm_projector_type != "mlp":
        raise ValueError(
            f"Merged report checkpoints require mm_projector_type='mlp', "
            f"got {config.mm_projector_type!r}"
        )
    return MultiModalProjector(
        input_size=config.mm_hidden_size,
        output_size=config.hidden_size,
        mlp_depth=config.mm_mlp_depth,
        proj_out_num=config.proj_out_num,
        out_norm=getattr(config, "mm_projector_norm", False),
    )
