"""CARE encoder for volume-level classification fine-tuning."""
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from model_loading.models import get_model
from models.lora import inject_lora, is_lora_param_name


def lr_factor(position, start, epochs):
    elapsed = position - start
    if elapsed < 0:
        return 0.
    warmup = 3 if start == 0 else 1
    if elapsed < warmup:
        return .1 + .9 * elapsed / warmup
    progress = min(1., (elapsed - warmup) / max(1, epochs - start - warmup))
    return .05 + .95 * .5 * (1 + math.cos(math.pi * progress))


def parameter_schedule(name, config, depth=32):
    fast = config.get('schedule_mode') == 'fast'
    if name.startswith(('head.', 'pooler.')):
        return 0, config['head_lr']
    if name.startswith(('visual.ln_post.', 'visual.proj')):
        return (0 if config['arm'] == 'R0' else 1 if fast else 3), config['encoder_lr']
    if config['lora_blocks']:
        return (1 if fast else 3, config['lora_lr']) if is_lora_param_name(name) else None
    prefix = 'visual.transformer.resblocks.'
    if not name.startswith(prefix):
        return None
    index = int(name[len(prefix):].split('.')[0])
    if index < depth - config['last_n']:
        return None
    start = 0 if config['arm'] == 'R0' else 3 if index >= depth - 2 else 6
    if fast:
        start = 1 if index >= depth - 4 else 3
    return start, config['encoder_lr'] * config['llrd'] ** (depth - 1 - index)


class TokenAttentionPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        with torch.no_grad():
            self.value.weight.copy_(torch.eye(width))
            self.value.bias.zero_()
        self.scale = width ** -.5

    def forward(self, tokens):
        logits = (self.key(tokens) @ self.query.to(tokens.dtype)) * self.scale
        weights = logits.softmax(dim=-1)
        return torch.einsum('bn,bnc->bc', weights, self.value(tokens))


class TuningModel(nn.Module):
    def __init__(self, config, pretrained, vendor=None, inject=True):
        super().__init__()
        if isinstance(pretrained, Mapping):
            care_checkpoint = pretrained['path']
            sha256 = pretrained.get('sha256')
        else:
            care_checkpoint = pretrained
            sha256 = None
        care_model = get_model('cpu', care_checkpoint)
        self.visual = care_model.visual
        self.weight_audit = dict(
            sha256=sha256, matched_tensors=len(self.visual.state_dict()), exact=True)
        del care_model
        conv = self.visual.conv1
        if conv.weight.shape[1] != 1:
            raise ValueError('Expected one-channel pretrained patch embedding')
        new = nn.Conv3d(3, conv.out_channels, conv.kernel_size, stride=conv.stride,
                        padding=conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            new.weight.copy_(conv.weight.repeat(1, 3, 1, 1, 1) / 3)
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        self.visual.conv1 = new
        self.config = config
        self.stage_epoch = 0
        self.lora_injected = False
        with torch.random.fork_rng():
            torch.manual_seed(config['seed'] + 101)
            outputs = len(config['labels']) + len(config.get('aux_labels') or [])
            if config.get('head_type', 'mlp') == 'ln_linear':
                self.head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, outputs))
            else:
                self.head = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True),
                                          nn.Dropout(.3), nn.Linear(512, outputs))
            if config.get('pool_type', 'tok') == 'attn':
                self.pooler = TokenAttentionPool(self.visual.transformer.width)
        for param in self.parameters():
            param.requires_grad = False
        if inject:
            self.add_adapters()
        # Non-reentrant checkpointing is already supported by the pinned ViT.
        self.visual.set_grad_checkpointing(True)

    def add_adapters(self):
        if not self.config['lora_blocks'] or self.lora_injected:
            return
        with torch.random.fork_rng():
            torch.manual_seed(self.config['seed'] + 202)
            count = self.config['lora_blocks']
            inject_lora(self.visual, list(range(32-count, 32)), {'qkv', 'out'},
                        self.config['rank'], 2*self.config['rank'], .05)
        self.lora_injected = True

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        if mode and self.config.get('train_droppath', False):
            for index, block in enumerate(self.visual.transformer.resblocks):
                if any(parameter_schedule(f'visual.transformer.resblocks.{index}.{name}',
                                          self.config) is not None
                       for name, _ in block.named_parameters()):
                    block.train(True)
        adapter_start = 1 if self.config.get('schedule_mode') == 'fast' else 3
        if mode and self.config['lora_blocks'] and getattr(self, 'stage_epoch', 0) >= adapter_start:
            for name, module in self.visual.named_modules():
                if name.endswith('lora_dropout'):
                    module.train(True)
        return self

    def optimizer(self):
        groups = {}
        for name, param in self.named_parameters():
            schedule = parameter_schedule(name, self.config)
            if schedule is None:
                continue
            start, peak = schedule
            decay = 0. if param.ndim <= 1 or name.endswith('.bias') else self.config['weight_decay']
            key = (start, peak, decay)
            group = groups.setdefault(key, dict(params=[], names=[], start=start, peak=peak,
                                                lr=0., weight_decay=decay))
            group['params'].append(param)
            group['names'].append(name)
        self.delta_names = {n for group in groups.values() for n in group['names']}
        return torch.optim.AdamW(list(groups.values()), lr=self.config['head_lr'])

    def set_stage(self, optimizer, epoch):
        self.stage_epoch = epoch
        for group in optimizer.param_groups:
            for param in group['params']:
                param.requires_grad = epoch >= group['start']
                if not param.requires_grad:
                    param.grad = None
        self.update_lrs(optimizer, epoch)
        return {f'start{start}': sum(p.numel() for g in optimizer.param_groups if g['start'] == start
                                    for p in g['params'] if p.requires_grad)
                for start in sorted({g['start'] for g in optimizer.param_groups})}

    def update_lrs(self, optimizer, position):
        for group in optimizer.param_groups:
            group['lr'] = group['peak'] * lr_factor(position, group['start'], self.config['epochs'])

    def delta(self):
        return {name: param.detach().cpu().clone() for name, param in self.named_parameters()
                if name in self.delta_names}

    def restore(self, delta):
        if set(delta) != self.delta_names:
            raise ValueError('Delta scope differs from scheduled parameters')
        with torch.no_grad():
            params = dict(self.named_parameters())
            for name, tensor in delta.items():
                if params[name].shape != tensor.shape or not torch.isfinite(tensor).all():
                    raise ValueError(f'Invalid delta: {name}')
                params[name].copy_(tensor)

    TOPK = 16
    LOGIT_POOLS = ('logit_topk', 'logit_maxmean', 'block_logit_maxmean')
    BLOCK_INDICES = (23, 31)

    def _topk_mean(self, tokens, k=None):
        k = min(self.TOPK if k is None else k, tokens.shape[1])
        return tokens.topk(k, dim=1).values.mean(dim=1)

    def _token_features(self, image):
        x = self.visual.transformer(self.visual._embeds(image))
        x = self.visual.ln_post(x)
        return x[:, 1:] @ self.visual.proj

    def _multiblock_tokens(self, image):
        x = self.visual._embeds(image)
        wanted, collected = set(self.BLOCK_INDICES), []
        for index, block in enumerate(self.visual.transformer.resblocks):
            x = block(x)
            if index in wanted:
                collected.append(self.visual.ln_post(x)[:, 1:] @ self.visual.proj)
        if len(collected) != len(wanted):
            raise ValueError(f'BLOCK_INDICES {self.BLOCK_INDICES} out of range for this tower')
        return sum(collected) / len(collected)

    def _logit_pool(self, image, pool):
        tokens = (self._multiblock_tokens(image) if pool == 'block_logit_maxmean'
                  else self._token_features(image))
        if self.config.get('feature_l2', True):
            tokens = F.normalize(tokens, dim=-1)
        logits = self.head(tokens)
        if pool == 'logit_topk':
            return self._topk_mean(logits)
        return .5 * logits.mean(dim=1) + .5 * self._topk_mean(logits)

    def pooled_features(self, image):
        pool = self.config.get('pool_type', 'tok')
        if pool == 'tok':
            features = self.visual(image)
            return features[0] if isinstance(features, tuple) else features
        if pool not in ('mean', 'attn', 'topk', 'maxmean'):
            raise ValueError(f'Unknown pool_type: {pool}')
        x = self.visual.transformer(self.visual._embeds(image))
        x = self.visual.ln_post(x)
        tokens = x[:, 1:]
        if pool == 'mean':
            pooled = tokens.mean(dim=1)
        elif pool == 'attn':
            pooled = self.pooler(tokens)
        elif pool == 'topk':
            pooled = self._topk_mean(tokens)
        else:
            pooled = .5 * tokens.mean(dim=1) + .5 * self._topk_mean(tokens)
        return pooled @ self.visual.proj

    def forward(self, image):
        if image.shape[1:] != (3, 112, 336, 336):
            raise ValueError(f'Unexpected Ours input shape: {tuple(image.shape)}')
        pool = self.config.get('pool_type', 'tok')
        if pool in self.LOGIT_POOLS:
            return self._logit_pool(image, pool)
        features = self.pooled_features(image)
        if self.config.get('feature_l2', True):
            features = F.normalize(features, dim=-1)
        return self.head(features)
