# Training

This page explains how to train PDE-Transformer models using the provided training scripts and configuration system.

## Running Training

The main training script is `main.py`. Here's how to use it:

```bash
python main.py -c config/ape_2d/supervised/pde-mc-s-mse.yaml --name my_experiment
```

### Command Line Arguments

The script supports several command line arguments:

| Argument | Description |
|----------|-------------|
| `-n, --name` | Name of the experiment (used for logging) |
| `--dryrun` | We use wandb for logging. If set, do not sync with the cloud|
| `-c, --config` | Path(s) to base config file(s). Can specify multiple configs that are loaded left-to-right |
| `-s, --seed` | Random seed for reproducibility |
| `-l, --logdir` | Base directory for logging (defaults to `logs/`) |
| `--env` | Path to environment config (defaults to `env/local.yaml`) |
| `--debug` | Enable debug mode |
| `--no-train` | Skip training (useful for testing only) |
| `--ema` | Use EMA weights for testing |
| `--no-inference` | Skip inference/tests after training |

## Configuration System

PDE-Transformer uses a YAML-based configuration system with OmegaConf. Configurations can be loaded from YAML files, 
modified via the command line or composed from multiple files.

### Example Configuration

Let's break down the `pde-mc-s-mse.yaml` configuration:

```yaml
# basic experiment settings
project: ape-transformers-mse
name: pde-mc-s-mse

batch_size: 8
max_epochs: 100

trainer:
    base_learning_rate: 4.0e-5
    batch_size: ${batch_size}  # References the batch_size defined above
    scale_lr: False
    params:
        _file: config/default_trainer_args.yaml  # Includes another config file
        max_epochs: ${max_epochs}
        accumulate_grad_batches: 8
        precision: bf16-mixed

# Model architecture
model:
    target: pdetransformer.core.mixed_channels.SingleStepSupervised
    params:
        model:
          target: pdetransformer.core.mixed_channels.PDETransformer
          params:
            sample_size: 256
            in_channels: 2
            out_channels: 2
            type: PDE-S
            patch_size: ${patch_size}
            periodic: True
            carrier_token_active: False

unrolling_steps: 1
test_unrolling_steps: 29

data:
  _file: config/ape_2d/data/multi_task_norm.yaml

frequency_log_images: 100
frequency_log_metrics: 100

callbacks:
  _file: config/default_callbacks.yaml
  ema:
    target: pdetransformer.callback.EMA
    params:
      decay: 0.999
  images:
    target: pdetransformer.callback.MultiTaskVideoLoggerCustom
    params:
      frequency: ${frequency_log_images}
      num_frames: 29
      num_inference_steps: 40
      test_only: True
      prepare_plots:
        target: pdetransformer.data.pbdl_module.prepare_plots
  simulation:
    target: pdetransformer.callback.Simulation2DMetricLoggerCustom
    params:
      frequency: ${frequency_log_metrics}
      batch_size: ${batch_size}
      num_frames: 29
      num_inference_steps: 40
      test_only: True
      metric_config:
        _file: config/default_simulation_metrics.yaml
  emagrad:
    target: pdetransformer.callback.EmaGradClip
    params:
      ema_coef1: 0.9
      ema_coef2: 0.99
      max_norm_ratio: 2.0
      clip_norm_ratio: 1.1
```

### Key Configuration Components

General instantiation of class. `target`: The Python class to instantiate. `params`: Parameters passed to the class constructor. Nested configurations for complex models are can be used.

2. **Training Parameters**
    - Learning rate (`base_learning_rate`)
    - Batch size (`batch_size`)
    - Number of epochs (`max_epochs`)
    - Precision
    - Gradient accumulation (`accumulate_grad_batches`)
    - Number of GPUs, trainer strategy, number of nodes, see `config/default_trainer_args.yaml`

1. **Model Configuration**
    - Instantiates the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) module as given by the `target` and `params`. Here we used autoregressive training with a single step into the future.
    - We use PDETransformer with a mixed channel representation `pdetransformer.mixed_channels.PDETransformer` with the small size S.


3. **Data Configuration**
    - See the file `config/ape_2d/data/multi_task_norm.yaml`
    - Dataset class `pdetransformer.data.MultiDataModule` combining different datasets and defining the train/val/test splits

4. **Callbacks** Different callbacks that define hooks (e.g. `on_train_epoch_end`)
    - EMA (Exponential Moving Average)
    - Image/Video logging
    - Metric logging
    - Gradient clipping during training

### Configuration Inheritance

Example of including another config:

```yaml
data:
  _file: config/ape_2d/data/multi_task_norm.yaml
```

### Overriding Configurations

You can override any configuration value via command line:

```bash
python main.py batch_size=16 max_epochs=200 -c config/ape_2d/supervised/pde-mc-s-mse.yaml 
```

### Example Training Commands

#### Basic Training
```bash
python main.py -c config/ape_2d/supervised/pde-mc-s-mse.yaml
```

#### Debug Mode
```bash
python main.py -c config/ape_2d/supervised/pde-mc-s-mse.yaml --debug
```

#### Test Only
```bash
python main.py -c config/ape_2d/supervised/pde-mc-s-mse.yaml --no-train --ema
```

#### Custom Configuration
```bash
python main.py batch_size=16 max_epochs=200 -c config/ape_2d/supervised/pde-mc-s-mse.yaml --seed=42
```

!!! note 
    When overriding configuration values via command line, place them immediately after `main.py` and before the config file specification. For example:
    ```bash
    python main.py batch_size=16 max_epochs=200 -c config/ape_2d/supervised/pde-mc-s-mse.yaml
    ```
    Not:
    ```bash
    python main.py -c config/ape_2d/supervised/pde-mc-s-mse.yaml batch_size=16 max_epochs=200
    ```



For more details about specific model architectures and training setups, see the [Model Types](mixed_channel.md) documentation. 