import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ModelStructure:
    """
    Tracks the structure of neural network models for block-wise operations.
    Supports both CNN and Transformer architectures.
    """

    def __init__(self):
        self.blocks = []
        self.block_names = []
        self.block_shapes = {}
        self.total_params = 0

    def add_block(self, name: str, shape: Tuple[int, ...], num_params: int):
        """Add a block (e.g., conv layer, attention head) to the structure."""
        self.blocks.append({
            'name': name,
            'shape': shape,
            'num_params': num_params,
            'start_idx': self.total_params,
            'end_idx': self.total_params + num_params
        })
        self.block_names.append(name)
        self.block_shapes[name] = shape
        self.total_params += num_params

    def get_block_params(self, params_flat: torch.Tensor, block_name: str) -> torch.Tensor:
        """Extract parameters for a specific block from flattened parameter vector."""
        block = next(b for b in self.blocks if b['name'] == block_name)
        return params_flat[block['start_idx']:block['end_idx']]

    def set_block_params(self, params_flat: torch.Tensor, block_name: str, block_params: torch.Tensor):
        """Set parameters for a specific block in flattened parameter vector."""
        block = next(b for b in self.blocks if b['name'] == block_name)
        params_flat[block['start_idx']:block['end_idx']] = block_params.flatten()

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> 'ModelStructure':
        """
        Automatically builds block structure from a PyTorch model.
        CRITICAL FOR PEFT/LoRA: Explicitly ignores frozen parameters!
        """
        structure = cls()
        for name, param in model.named_parameters():
            if param.requires_grad:  # This perfectly isolates LoRA adapters
                structure.add_block(
                    name=name,
                    shape=param.shape,
                    num_params=param.numel()
                )
        return structure


class CIFARCNN(nn.Module):
    """CNN for CIFAR-10/100 with explicit block structure tracking."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.structure = self._build_structure()

    def _build_structure(self) -> ModelStructure:
        """Build block structure for this CNN."""
        structure = ModelStructure()

        # Conv blocks
        structure.add_block('conv1_weight', self.conv1.weight.shape, self.conv1.weight.numel())
        structure.add_block('conv1_bias', self.conv1.bias.shape, self.conv1.bias.numel())
        structure.add_block('conv2_weight', self.conv2.weight.shape, self.conv2.weight.numel())
        structure.add_block('conv2_bias', self.conv2.bias.shape, self.conv2.bias.numel())
        structure.add_block('conv3_weight', self.conv3.weight.shape, self.conv3.weight.numel())
        structure.add_block('conv3_bias', self.conv3.bias.shape, self.conv3.bias.numel())

        # FC blocks
        structure.add_block('fc1_weight', self.fc1.weight.shape, self.fc1.weight.numel())
        structure.add_block('fc1_bias', self.fc1.bias.shape, self.fc1.bias.numel())
        structure.add_block('fc2_weight', self.fc2.weight.shape, self.fc2.weight.numel())
        structure.add_block('fc2_bias', self.fc2.bias.shape, self.fc2.bias.numel())

        return structure

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_weights_flat(self) -> torch.Tensor:
        """Get all model parameters as a flat tensor."""
        params = []
        for param in self.parameters():
            params.append(param.flatten())
        return torch.cat(params)

    def set_weights_flat(self, params_flat: torch.Tensor):
        """Set all model parameters from a flat tensor."""
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = params_flat[idx:idx + param_size].reshape(param.shape)
            idx += param_size


class SimpleTransformer(nn.Module):
    """Simple Transformer for federated learning experiments."""

    def __init__(self, vocab_size: int = 10000, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(d_model, num_classes)
        self.structure = self._build_structure()

    def _build_structure(self) -> ModelStructure:
        """Build block structure for this Transformer."""
        structure = ModelStructure()

        # Embedding block
        structure.add_block('embedding', self.embedding.weight.shape, self.embedding.weight.numel())
        structure.add_block('pos_encoding', self.pos_encoding.shape, self.pos_encoding.numel())

        # Transformer blocks (attention heads)
        for i, layer in enumerate(self.transformer.layers):
            # Multi-head attention
            structure.add_block(f'layer_{i}_self_attn_in_proj_weight',
                              layer.self_attn.in_proj_weight.shape,
                              layer.self_attn.in_proj_weight.numel())
            structure.add_block(f'layer_{i}_self_attn_out_proj_weight',
                              layer.self_attn.out_proj.weight.shape,
                              layer.self_attn.out_proj.weight.numel())

            # Feed-forward blocks
            structure.add_block(f'layer_{i}_linear1_weight',
                              layer.linear1.weight.shape,
                              layer.linear1.weight.numel())
            structure.add_block(f'layer_{i}_linear2_weight',
                              layer.linear2.weight.shape,
                              layer.linear2.weight.numel())

        # Classifier block
        structure.add_block('classifier_weight', self.classifier.weight.shape, self.classifier.weight.numel())
        structure.add_block('classifier_bias', self.classifier.bias.shape, self.classifier.bias.numel())

        return structure

    def forward(self, x, attention_mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

    def get_weights_flat(self) -> torch.Tensor:
        """Get all model parameters as a flat tensor."""
        params = []
        for param in self.parameters():
            params.append(param.flatten())
        return torch.cat(params)

    def set_weights_flat(self, params_flat: torch.Tensor):
        """Set all model parameters from a flat tensor."""
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = params_flat[idx:idx + param_size].reshape(param.shape)
            idx += param_size


class FederatedModel(nn.Module):
    """Base class for federated learning models with TAVS-ESP support."""

    def get_weights_flat(self) -> torch.Tensor:
        """Get all TRAINABLE model parameters as a flat tensor."""
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param.flatten())

        if not params:
            return torch.tensor([])

        return torch.cat(params)

    def set_weights_flat(self, params_flat: torch.Tensor):
        """Set all TRAINABLE model parameters from a flat tensor."""
        idx = 0
        for param in self.parameters():
            if param.requires_grad:
                param_size = param.numel()
                param.data = params_flat[idx:idx + param_size].reshape(param.shape)
                idx += param_size


class _BasicBlock(nn.Module):
    """One ResNet BasicBlock, GroupNorm instead of BatchNorm."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 gn_groups: int = 32):
        super().__init__()

        def norm(c):
            return nn.GroupNorm(min(gn_groups, c), c)

        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.gn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.gn2 = norm(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                norm(planes),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 for 32x32 CIFAR, with GroupNorm and explicit block structure.

    TWO DELIBERATE DEVIATIONS FROM torchvision's ResNet-18, both forced:

    1. CIFAR STEM. torchvision's stem is a 7x7 stride-2 conv followed by a
       3x3 stride-2 maxpool, which is built for 224x224 ImageNet and throws
       away a 32x32 image before the first residual stage. The standard
       CIFAR variant -- used by FedCorr, FedGR and FNBench alike -- is a
       3x3 stride-1 conv with no maxpool. Same parameter count to within
       the stem.

    2. GROUPNORM, NOT BATCHNORM. This one is not stylistic: BatchNorm would
       silently break this codebase. Both the client/server transport and
       the server-side evaluate_fn move weights with
       `zip(model.parameters(), ndarrays)`, and BatchNorm's running_mean and
       running_var are BUFFERS, not parameters. They would never leave the
       client, so the global model used for evaluation would keep its
       freshly-initialised statistics (mean 0, var 1) for the whole run and
       score near chance -- which would read as "ResNet-18 does not work
       here" when the real fault is transport. GroupNorm has no buffers at
       all (verified: `named_buffers()` is empty), so every one of its
       parameters is communicated. This is also the standard fix in the FL
       literature for normalisation under non-IID clients.

    The block structure is registered per parameter tensor, matching
    CIFARCNN's convention, so BVD scores each tensor separately. 62 blocks
    and 11.17M parameters; the ESP projection matrices total 8.1 GB per
    round against the CIFARCNN's existing 7.4 GB, so ESP needs no change.
    """

    def __init__(self, num_classes: int = 10, gn_groups: int = 32):
        super().__init__()
        self.in_planes = 64
        self.gn_groups = gn_groups
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(min(gn_groups, 64), 64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)
        self.structure = self._build_structure()

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        layers = []
        for st in [stride] + [1] * (num_blocks - 1):
            layers.append(_BasicBlock(self.in_planes, planes, st,
                                      self.gn_groups))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _build_structure(self) -> ModelStructure:
        """One block per parameter tensor, named like CIFARCNN's.

        Built by walking named_parameters() rather than listing tensors by
        hand: a hand-written list silently drifts the moment the
        architecture changes, and a block registered with the wrong size
        would corrupt every ESP projection that uses it.
        """
        structure = ModelStructure()
        for name, param in self.named_parameters():
            structure.add_block(name.replace(".", "_"), tuple(param.shape),
                                param.numel())
        return structure

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def get_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create models."""
    if model_type == 'cifar_cnn':
        return CIFARCNN(**kwargs)
    elif model_type == 'resnet18':
        return ResNet18(**kwargs)
    elif model_type == 'simple_transformer':
        return SimpleTransformer(**kwargs)
    elif model_type == 'gpt2' or model_type == 'gpt2_small':
        # Import GPT-2 model here to avoid circular imports
        from .gpt2_model import get_gpt2_model
        return get_gpt2_model(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")