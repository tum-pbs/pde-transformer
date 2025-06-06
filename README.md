# PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations

<div align="center">

<p align="center">
<a href="https://pypi.org/project/pdetransformer/">
  <img src="https://img.shields.io/pypi/v/pretransformer.svg" alt="PyPI">
</a>
<!--- 
<a href="https://github.com/tum-pbs/pde-transformer/actions/workflows/test.yml">
  <img src="https://github.com/tum-pbs/pde-transformer/actions/workflows/test.yml/badge.svg" alt="Tests">
</a>
---!>
<a href="https://tum-pbs.github.io/pde-transformer">
  <img src="https://img.shields.io/badge/docs-latest-green" alt="docs-latest">
</a>
<a href="https://github.com/ceyron/apebench/releases">
  <img src="https://img.shields.io/github/v/release/tum-pbs/pde-transformer?include_prereleases&label=changelog" alt="Changelog">
</a>
<a href="https://github.com/ceyron/apebench/blob/main/LICENSE.txt">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</a>
</p>

[Paper](https://arxiv.org/pdf/2505.24717.pdf) • 
[Project Page](https://tum-pbs.github.io/pde-transformer/landing.html) • 
[🤗 Hugging Face Models](https://huggingface.co/thuerey-group/pde-transformer)
<br>
[📦 Installation](#-quick-installation) •
[📝 Description](#-model-description) •
[🎯 Features](#-key-highlights) •
[🔬 PDE Types](#-supported-pde-types)

</div>

---

**Authors:** [Benjamin Holzschuh](https://holzschuh.github.io/), [Qiang Liu](https://qiangliu-ai.github.io/), [Georg Kohl](https://georgkohl.github.io/), [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/)

**PDE-Transformer** is a state-of-the-art neural architecture for physics simulations, specifically designed for partial differential equations (PDEs). This work will be presented at **ICML 2025** and represents a significant advancement in neural PDE solvers with transformer architectures.

### 🎯 Key Highlights
- **🔧 Production Ready**: Available on Test PyPI for easy installation and experimentation
- **📈 State-of-the-Art**: Outperforms existing methods across 16 different PDE types
- **🌐 Open Source**: Full implementation with pre-trained models and comprehensive documentation

### 📦 Quick Installation

```bash
# Install from Test PyPI
pip install -i https://test.pypi.org/simple/ pdetransformer

# Or install from source
git clone https://github.com/pde-transformer/pde-transformer.git
cd pde-transformer
pip install -e .
```

## 📝 Model Description

PDE-Transformer is designed to efficiently process and predict the evolution of physical systems described by partial differential equations (PDEs). 
It can handle multiple types of PDEs, different resolutions, domain extents, boundary conditions, 
and includes deep conditioning mechanisms for PDE- and task-specific information.

Key features:
- **Multi-scale architecture** with token down- and upsampling for efficient modeling
- **Shifted window attention** for improved scaling to high-resolution data
- **Mixed Channel (MC) and Separate Channel (SC)** representations for handling multiple physical quantities
- **Flexible conditioning mechanism** for PDE parameters, boundary conditions, and simulation metadata
- **Pre-training and fine-tuning capabilities** for transfer learning across different physics domains

### Training Objectives

The model supports both supervised and diffusion training:

- **Supervised Training**: Direct MSE loss for deterministic, unique solutions
- **Flow Matching**: For probabilistic modeling and uncertainty quantification

## 🎯 Supported PDE Types

PDE-Transformer has been trained and evaluated on 16 different types of PDEs including:

- **Linear PDEs**: Diffusion
- **Nonlinear PDEs**: Burgers, Korteweg-de-Vries, Kuramoto-Sivashinsky
- **Reaction-Diffusion**: Fisher-KPP, Swift-Hohenberg, Gray-Scott
- **Fluid Dynamics**: Navier-Stokes (Decaying Turbulence, Kolmogorov Flow)

## 🚀 Usage

### Quick Start

```python
from pdetransformer import PDETransformer

# Load pre-trained model
model = PDETransformer.from_pretrained("pde-transformer-sc-b")

# For physics simulation
predictions = model(x)
```

## 📚 Citation

If you use PDE-Transformer in your research, please cite:

```bibtex
@article{holzschuh2024pde,
  title={PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations},
  author={Holzschuh, Benjamin and Liu, Qiang and Kohl, Georg and Thuerey, Nils},
  booktitle={Forty-second International Conference on Machine Learning, {ICML} 2025, Vancouver, Canada, July 13-19, 2025},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

---

**Note**: This is a research project from the Technical University of Munich (TUM) Physics-based Simulation Group. 
For questions and support, please refer to the GitHub repository or contact the authors.