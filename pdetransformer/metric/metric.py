from omegaconf import DictConfig
from ..utils import instantiate_from_config
from torchmetrics import Metric

def get_metrics(config: DictConfig):
    metrics = []
    for metric_key in config:
        metrics.append(instantiate_from_config(config[metric_key]))
    return metrics