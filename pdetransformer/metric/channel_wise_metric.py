from torchmetrics import Metric
from typing import Optional, List, Dict, Callable, Any

from torch import Tensor
from pdetransformer.utils import instantiate_from_config

import os

class ChannelWiseMetric(Metric):
    def __init__(self, num_fields: int, metric_args: Dict, field_names: Optional[List[str]] = None,
                 **kwargs):

        super().__init__(**kwargs)

        self.num_fields = num_fields

        if field_names is None:
            field_names = [f"field_{i}" for i in range(num_fields)]
        self.field_names = field_names

        self.field_metrics: List[Metric] = []
        for i in range(num_fields):
            self.field_metrics.append(
                instantiate_from_config(metric_args))


    def reset(self) -> None:
        """
        Reset the internal state of the metric.
        """

        for field_metric in self.field_metrics:
            field_metric.reset()

        super().reset()

    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor) -> None:

        for i, field_metric in enumerate(self.field_metrics):
            field_preds = preds[:, :, i:i+1]
            field_target = target[:, :, i:i+1]
            field_metric.update(field_preds, field_target, class_labels)

    def compute(self) -> Tensor:
        result: List[Tensor] = []
        for field_metric in self.field_metrics:
            result.append(field_metric.compute())

        return Tensor(result)

    def save(self, path, file):

        # create path if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        for i, field_metric in enumerate(self.field_metrics):
            field_file = f'{self.field_names[i]}_{file}'

            field_metric.save(os.path.join(path, field_file))

    def persistent(self, mode: bool = False) -> None:
        """Change if metric state is persistent (save as part of state_dict) or not.

        Args:
            mode: bool indicating if all states should be persistent or not

        """
        for field_metric in self.field_metrics:
            field_metric.persistent(mode=mode)

    def __repr__(self) -> str:
        """Return a representation of the compositional metric, including the two inputs it was formed from."""
        _op_metrics = f"(\n  {',\n  '.join([repr(metric) for metric in self.field_metrics])}\n)"
        return f"{self.__class__.__name__}(\n  num_fields={self.num_fields},\n  field_names={self.field_names},\n  field_metrics={_op_metrics}\n)"

    def _wrap_compute(self, compute: Callable) -> Callable:
        """No wrapping necessary for compositional metrics."""
        return compute

import torch.jit

@torch.jit.unused
def forward(self, *args: Any, **kwargs: Any) -> Any:
    """Calculate metric on current batch and accumulate to global state."""
    val_list = []
    for field_metric in self.field_metrics:
        val = field_metric(*args, **kwargs)
        val_list.append(val)

    self._forward_cache = val_list
    return self._forward_cache
