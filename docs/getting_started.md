---
layout: default
title: Getting Started
---

# Getting Started

This guide will help you get started with PDE-Transformer, from installation to running your first physics simulation.

## Installation

You can install PDE-Transformer using pip:

```bash
# Install from Test PyPI
pip install pdetransformer

# Or install from source
git clone https://github.com/pde-transformer/pde-transformer.git
cd pde-transformer
pip install -e .
```

## Quick Usage Example

Here's a simple example to get you started with PDE-Transformer. This loads a pretrained model from [Hugging Face](https://huggingface.co/thuerey-group/pde-transformer).

```python
from pdetransformer.core.mixed_channels import PDETransformer
import torch

# Load pre-trained model
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').cuda()

# For physics simulation
x = torch.randn((1,2,256,256), dtype=torch.float32).cuda() # batch x channels x height x width
predictions = model(x)
```

Alternatively, we can initialize the model from scratch

```python
model = PDETransformer(
    sample_size = 256,
    in_channels = 2,
    out_channels = 2,
    type = 'PDE-S',
)
```

## Model Variants

PDE-Transformer comes in two main variants:

1. **Mixed Channel (MC)**: Processes all physical quantities together. This is what we have been using in the quick usage example. See [Mixed Channels](mixed_channel.md) for details.
2. **Separate Channel (SC)**: Processes each physical quantity separately. See [Separate Channels](separate_channel.md) for details.

Choose the appropriate variant based on your specific use case.

## Pre-trained Models

We provide several pre-trained models that you can use out of the box. See our [Pretrained Models](pretrained-weights.md) page for more details.
