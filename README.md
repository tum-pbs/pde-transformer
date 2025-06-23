# PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations

<div align="center">

<p align="center">
<a href="https://pypi.org/project/pdetransformer/">
  <img src="https://img.shields.io/pypi/v/pretransformer.svg" alt="PyPI">
</a> 
<a href="https://tum-pbs.github.io/pde-transformer">
  <img src="https://img.shields.io/badge/docs-latest-green" alt="docs-latest">
</a>
<a href="https://github.com/tum-pbs/pde-transformer/releases">
  <img src="https://img.shields.io/github/v/release/tum-pbs/pde-transformer?include_prereleases&label=changelog" alt="Changelog">
</a>
<a href="https://github.com/tum-pbs/pde-transformer/blob/main/LICENSE.txt">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</a>
</p>

[Paper](https://arxiv.org/pdf/2505.24717.pdf) • 
[Project Page](https://tum-pbs.github.io/pde-transformer/landing.html) • 
[🤗 Hugging Face](https://huggingface.co/thuerey-group/pde-transformer) • 
[Documentation](https://tum-pbs.github.io/pde-transformer)
<br>
[Installation](#-quick-installation) •
[Description](#-model-description) •
[Features](#-key-highlights) •
[Usage](#-usage)

</div>

---

**Authors:** [Benjamin Holzschuh](), [Qiang Liu](), [Georg Kohl](), [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/)

**PDE-Transformer** is a state-of-the-art neural architecture for physics simulations, specifically designed for partial differential equations (PDEs) on regular grids. This work will be presented at **ICML 2025**.

### Key Highlights
- **Production Ready**: Available as a pip package for easy installation and experimentation.
- **State-of-the-Art**: Outperforms existing methods across 16 different types of PDEs and three challenging downstream tasks involving complex dynamics. 
- **Transfer Learning**: Improved performance when adapting pre-trained models to new physics problems with limited training data.
- **Open Source**: Full implementation with pre-trained models and comprehensive documentation.

### Quick Installation

```bash
# Install from Test PyPI
pip install pdetransformer

# Or install from source
git clone https://github.com/pde-transformer/pde-transformer.git
cd pde-transformer
pip install -e .
```

## Model Description

PDE-Transformer is designed to efficiently process and predict the evolution of physical systems described by partial differential equations (PDEs). It can handle multiple types of PDEs, different resolutions, domain extents, boundary conditions, 
and includes deep conditioning mechanisms for PDE- and task-specific information.

Key features:
- **Multi-scale architecture** with token down- and upsampling for efficient modeling.
- **Shifted window attention** for improved scaling to high-resolution data.
- **Mixed Channel (MC) and Separate Channel (SC)** representations for handling multiple physical quantities.
- **Flexible conditioning mechanism** for PDE parameters, boundary conditions, and simulation metadata.
- **Pre-training and fine-tuning capabilities** for transfer learning across different physics domains.

### Training Objectives

The model supports both supervised and diffusion training:

- **Supervised Training**: Direct MSE loss for deterministic, unique solutions. Fast training and inference. 
- **Flow Matching**: For probabilistic modeling and uncertainty quantification.

## Supported PDE Types

PDE-Transformer has been trained and evaluated on 16 different types of PDEs including:

- **Linear PDEs**: Diffusion
- **Nonlinear PDEs**: Burgers, Korteweg-de-Vries, Kuramoto-Sivashinsky
- **Reaction-Diffusion**: Fisher-KPP, Swift-Hohenberg, Gray-Scott
- **Fluid Dynamics**: Navier-Stokes (Decaying Turbulence, Kolmogorov Flow)

## Usage

### Quick Start

```python
from pdetransformer.core.mixed_channels import PDETransformer
import torch

# Load pre-trained model
model = PDETransformer.from_pretrained('thuerey-group/pde-transformer', subfolder='mc-s').cuda()

# For physics simulation
x = torch.randn((1,2,256,256), dtype=torch.float32).cuda()
predictions = model(x)
```

## Documentation

For detailed documentation, visit [tum-pbs.github.io/pde-transformer](https://tum-pbs.github.io/pde-transformer/).

## Citation

If you use PDE-Transformer in your research, please cite:

```bibtex
@article{holzschuh2024pde,
  title={PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations},
  author={Holzschuh, Benjamin and Liu, Qiang and Kohl, Georg and Thuerey, Nils},
  booktitle={Forty-second International Conference on Machine Learning, {ICML} 2025, Vancouver, Canada, July 13-19, 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Note**: This is a research project from the Technical University of Munich (TUM) Physics-based Simulation Group. 
For questions and support, please refer to the GitHub repository or contact the authors.
