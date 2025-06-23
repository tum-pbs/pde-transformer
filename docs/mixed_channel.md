# Mixed Channels Version of PDE-Transformer

The mixed channels (MC) version of PDE-Transformer is an efficient implementation that embeds different physical channels within the same token. This approach offers several advantages and trade-offs compared to the separate channels version.

## Key Characteristics

### Architecture
- Embeds different physical channels (e.g., velocity, density) within the same token
- Uses a shared token representation for all physical quantities
- Maintains a single attention mechanism across all channels
### Advantages
1. **Computational Efficiency**
    - Higher information density per token
    - Reduced memory footprint
    - Faster training and inference times
    - More efficient use of model parameters
2. **Simplified Architecture**
    - Single token stream for all physical quantities
    - Simpler implementation and maintenance
### Limitations
1. **Transfer Learning Constraints**
    - Channel types must be known at training time
    - Less flexible for transfer learning applications
    - May require retraining for new channel configurations

2. **Representation Learning**
    - Less disentangled representation of physical channels
    - May require more training data to learn complex relationships

## Performance

The mixed channels version achieves comparable performance to the separate channels version while being more computationally efficient. However, it may show reduced performance in transfer learning scenarios or when dealing with novel channel configurations.

## Initialization

Here's how to initialize the PDETransformer model:

```python
from pdetransformer.core.mixed_channels import PDETransformer

# Initialize the model
model = PDETransformer(
    sample_size=256,         
    in_channels=4,          
    out_channels=4,        
    type="PDE-S",          
    periodic=True,       
    carrier_token_active=False,
    window_size=8,       
    patch_size=4,          
)

# Optional parameters (kwargs)
model = PDETransformer(
    # ... required parameters ...
    hidden_size=96,        
    num_heads=4,            
    mlp_ratio=4.0,          
    class_dropout_prob=0.1, 
    num_classes=1000,     
)
```
### Parameter Explanation
- `sample_size`: The spatial dimensions of the input/output grid (e.g., 64 for a 64x64 grid). This is a positional argument, but it not used right now and only there to provide a unified initialization of different models. Therefore it can be set to an arbitrary value and the model can be applied to grids of variable sizes.
- `in_channels`: Number of input physical channels (e.g., velocity components, density)
- `out_channels`: Number of output physical channels
- `type`: Defines the model config, i.e. PDE-S, PDE-B or PDE-L
- `periodic`: Whether to use periodic boundary conditions in the simulation
- `carrier_token_active`: Whether to use carrier tokens for global information exchange. This enables hierarchical attention (https://arxiv.org/abs/2306.06189). Not compatible with periodic boundary conditions at the moment; default: False. 
- `window_size`: Size of the local attention window (larger windows capture more context but require more computation)
- `patch_size`: Size of patches used for token embedding (smaller patches preserve more spatial detail but increase computation)
- `hidden_size`: Dimension of the hidden representations (determines model capacity)
- `num_heads`: Number of attention heads (more heads allow for different types of attention patterns)
- `mlp_ratio`: Ratio of MLP hidden dimension to embedding dimension
- `class_dropout_prob`: Dropout probability for class conditioning
- `num_classes`: Number of classes for conditioning (e.g., different PDE types or simulation parameters)

## Usage 

Here's an example of how to use the model for forward prediction:

```
import torch

model = model.cuda()

# prepare input data
batch_size = 4
height, width = 64, 64
num_channels = 4

x = torch.randn(batch_size, num_channels, height, width).cuda()

class_labels = torch.zeros(batch_size).int().cuda()

timestep = torch.zeros(batch_size).cuda()

output = model(
    hidden_states=x,
    timestep=timestep,
    class_labels=class_labels,
)

print('Output shape: ', output.sample.shape)
```

### Forward Pass Parameters

- `x`: tensor of shape (B, C, H, W)
    - B: Batch size
    - C: Input channels
    - H, W: Height and width of the spatial grid

- `timestep`: tensor of shape (B,). Can be set to `None`. Used as diffusion time when training probabilistic models.

- `class_labels`: tensor of shape (B,). Needs to be int or long. Can be set to `None`. Used to indicate type of PDE, see [datasets](datasets/ape_2d.md#simulation-type-label) for class labels of APE 2D PDEs.

### Example Notebook

An example notebook how to run inference for pretrained PDE-Transformer and additional explanations/code examples can be found at 
[notebooks/visualization_mc_ape2d.ipynb](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/notebooks/visualization_mc_ape2d.ipynb).