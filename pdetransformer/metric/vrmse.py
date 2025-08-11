import os
from typing import Optional, List

from torch import Tensor
from torchmetrics import Metric
import numpy as np
import torch
import pandas as pd

class VRMSE(Metric):

    def __init__(self, num_tasks: int, num_spatial_dims: int, max_length: int = 100,
                 dataset_names: Optional[List[str]] = None, field_name: str = 'field_0', **kwargs):

        super().__init__(**kwargs)

        self.times = list(range(max_length))
        self.num_tasks = num_tasks
        self.max_length = max_length
        self.num_spatial_dims = num_spatial_dims
        self.num_fields = 1
        self.field_name = field_name

        if dataset_names is None:
            dataset_names = [f"dataset_{i}" for i in range(num_tasks)]

        self.dataset_names = dataset_names

    def _update(self, metric: Tensor, class_labels: Tensor) -> None:
        """
        Update the internal state with the given metric and class labels
        :param metric: tensor of shape (batch_size, num_time_dim) containing the metric values
        :param class_labels: tensor of shape (batch_size,) containing the class labels
        """
        num_time_dim = len(self.times)
        filtered_times = [time for time in self.times if time < metric.shape[1]]

        index_tensor = torch.tensor(filtered_times).to(metric.device).unsqueeze(0).repeat(metric.shape[0], 1)
        metric = metric.gather(dim=1, index=index_tensor)

        metric = torch.nn.functional.pad(metric, (0, num_time_dim - metric.shape[1]))

        mse_init = torch.zeros_like(self.metric_sum)

        # update mse_sum at index from class_labels
        mse = mse_init.scatter_add(dim=0, index=class_labels.unsqueeze(1).expand(-1, metric.shape[1]),
                                   src=metric)

        self.metric_sum += mse

        update_count = torch.zeros_like(self.total)
        index_tensor = class_labels.unsqueeze(1).expand(-1, update_count.shape[1])
        src = torch.ones_like(index_tensor).float()

        update_count = update_count.scatter_add(dim=0,
                                                index=index_tensor,
                                                src=src)

        self.total += update_count

    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor, eps: float = 1e-7) -> None:
        """
        Update the metric with predictions and target values.
        :param preds: tensor of shape (batch_size, time, 1, spatial_dims...) containing the predictions
        :param target: tensor of shape (batch_size, time, 1, spatial_dims...) containing the target values
        :param class_labels: vector of shape (batch_size,) containing the class labels
        :param eps: small value to avoid division by zero
        :return:
        """

        mse = (preds - target) ** 2
        average_per_time = mse.mean(dim=tuple(range(-self.num_spatial_dims, 0)))
        mse_norm = (target - average_per_time) ** 2
        mse = mse * (1 / (mse_norm + eps))
        mse = torch.sqrt(mse)

        self._update(mse, class_labels)

    def compute(self) -> Tensor:

        result = self.metric_sum.float() / self.total
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        return result

    def save(self, path, file):

        # create path if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        metric = self.compute().T.cpu().numpy()

        # save metric as csv
        metric_df = pd.DataFrame(metric[:, :len(self.dataset_names)], columns=self.dataset_names)
        metric_df["time"] = self.times
        metric_df = metric_df.set_index("time")
        metric_df.to_csv(path + file)

        return metric_df
