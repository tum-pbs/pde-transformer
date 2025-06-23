# Pretrained Models

PDE-Transformer provides several pretrained models optimized for different use cases. This page details the available models and how to use them.

## Loading Pretrained Models

You can easily load any of our pretrained models using the following code:

```python
from pdetransformer.core.mixed_channels import PDETransformer
import torch

# Load pre-trained model
subfolder = 'mc-s'
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder=subfolder).cuda()

# For physics simulation
x = torch.randn((1,2,256,256), dtype=torch.float32).cuda()
predictions = model(x)
```

The model variant can be chosen via the subfolder, see the following list of pretrained models. In case you want to load a model of the separate channel variant, modify the import of PDETransformer to 

```python
from pdetransformer.core.separate_channels import PDETransformer
```

### Available Models

| Model | Channels | Size | Hidden Dim | Heads | Parameters | Training Epochs | Model Size |
|-------|----------|------|------------|-------|------------|----------------|------------|
| **sc-s** | Separate | Small | 96 | 4 | ~46M | 100 | ~133MB |
| **sc-b** | Separate | Base | 192 | 8 | ~178M | 100 | ~522MB |
| **sc-l** | Separate | Large | 384 | 16 | ~701M | 100 | ~2.07GB |
| **mc-s** | Mixed | Small | 96 | 4 | ~33M | 100 | ~187MB |
| **mc-b** | Mixed | Base | 192 | 8 | ~130M | 100 | ~716MB |
| **mc-l** | Mixed | Large | 384 | 16 | ~518M | 100 | ~2.81GB |

### Model Specifications of Pretrained Models

- **Separate Channel (SC)**: Embeds different physical channels independently with channel-wise axial attention. Number of input/outputs channels is variable.
- **Mixed Channel (MC)**: Embeds all physical channels within the same token representation. Using 2 input/output channels.
- **Patch Size**: Embeds 4×4 patch into spatio-temporal token. 
- **Window Size**: 8×8 for windowed attention
- **Boundary Conditions**: Supports both periodic and non-periodic boundary conditions

### Pretraining Datasets and Performance

The table below shows the performance differences using the nRMSE after 1 and 10 autoregressive steps on the [pretraining datasets](datasets/ape_2d.md).

| Model | Channels | Size | nRMSE1 | nRMSE10 |
|-------|----------|------|--------|---------|
| SC-S | Separate | Small | 0.043 | 0.34 |
| SC-B | Separate | Base | 0.037 | 0.29 |
| SC-L | Separate | Large | 0.034 | 0.26 |
| MC-S | Mixed | Small | 0.044 | 0.36 |
| MC-B | Mixed | Base | 0.038 | 0.31 |
| MC-L | Mixed | Large | 0.034 | 0.27 |




