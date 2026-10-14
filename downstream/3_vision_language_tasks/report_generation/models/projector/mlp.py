import torch.nn as nn


class MultiModalProjector(nn.Module):
    def __init__(self, input_size, output_size, mlp_depth, proj_out_num=33, out_norm=True):
        super().__init__()
        self.proj_out_num = proj_out_num
        self.mm_projector = nn.Sequential(
            nn.Linear(input_size, output_size),
            *[
                nn.Sequential(nn.GELU(), nn.Linear(output_size, output_size))
                for _ in range(mlp_depth - 1)
            ],
        )
        self.out_norm = nn.LayerNorm(output_size) if out_norm else None

    def forward(self, x):
        x = self.mm_projector(x)
        return self.out_norm(x) if self.out_norm is not None else x
