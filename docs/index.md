# PDE-Transformer Documentation

Welcome to the documentation for PDE-Transformer, a state-of-the-art neural architecture for physics simulations, specifically designed for partial differential equations (PDEs) on regular grids.

## Overview

PDE-Transformer is designed to efficiently process and predict the evolution of physical systems described by partial differential equations (PDEs). Our model provides:

- **Production Ready**: Available as a pip package for easy installation and experimentation
- **State-of-the-Art Performance**: Outperforms existing methods across a wide range different types of PDEs
- **Transfer Learning**: Improved performance when adapting pre-trained models to new physics problems
- **Open Source**: Full implementation with pre-trained models and documentation

## Key Features

### Architecture
- Multi-scale transformer architecture with token down- and upsampling for efficient modeling
- Shifted window attention for improved scaling to high-resolution data
- Mixed Channel (MC) and Separate Channel (SC) representations
- Flexible conditioning mechanism for PDE parameters and metadata

### Different Pretraining Datasets
- **Linear PDEs**: Diffusion
- **Nonlinear PDEs**: Burgers, Korteweg-de-Vries, Kuramoto-Sivashinsky
- **Reaction-Diffusion**: Fisher-KPP, Swift-Hohenberg, Gray-Scott
- **Fluid Dynamics**: Navier-Stokes (Decaying Turbulence, Kolmogorov Flow)

### Training Objectives
- **Supervised Training**: Direct MSE loss for deterministic, unique solutions
- **Flow Matching**: For probabilistic modeling and uncertainty quantification

## Code

The implementation is available on GitHub: [tum-pbs/pde-transformer](https://github.com/tum-pbs/pde-transformer)

```bash
# Install via pip
pip install pdetransformer

# Install from source
git clone https://github.com/tum-pbs/pde-transformer
cd pde-transformer
pip install -e .
```

For detailed documentation, see our [Documentation](getting-started.md).


## Citation

If you use PDE-Transformer in your research, please cite:

```bibtex
@article{holzschuh2025pde,
  title={PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations},
  author={Holzschuh, Benjamin and Liu, Qiang and Kohl, Georg and Thuerey, Nils},
  booktitle={Forty-second International Conference on Machine Learning, {ICML} 2025, Vancouver, Canada, July 13-19, 2025},
  year={2025}
}
```

## Acknowledgments

This work was supported by the ERC Consolidator Grant
SpaTe (CoG-2019-863850). The authors gratefully acknowledge the scientific support and resources of the AI service
infrastructure LRZ AI Systems provided by the Leibniz Supercomputing Centre (LRZ) of the Bavarian Academy of
Sciences and Humanities (BAdW), funded by Bayerisches
Staatsministerium fur Wissenschaft und Kunst (StMWK).