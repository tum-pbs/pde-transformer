import os
from typing import Optional, List

from torch import Tensor
from torchmetrics import Metric
import numpy as np
import torch
import pandas as pd

class SimulationMSE(Metric):
    def __init__(self, frequency: int, num_tasks: int, max_length: int = 100,
                 normalized: bool = True, dataset_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)

        self.frequency = frequency

        self.times = [2 ** i - 1 for i in range(int (np.log2(self.frequency)) + 1)]
        self.times.extend([(i + 1) * frequency - 1 for i in range(int (max_length // frequency))])

        self.num_tasks = num_tasks
        self.max_length = max_length
        self.normalized = normalized

        if dataset_names is None:
            dataset_names = [f"dataset_{i}" for i in range(num_tasks)]

        self.dataset_names = dataset_names

        self.add_state("metric_sum", default=torch.zeros(size=(num_tasks, len(self.times))),
                       dist_reduce_fx="sum")

        self.add_state("total", default=torch.zeros(size=(num_tasks, len(self.times))),
                       dist_reduce_fx="sum")

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

    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor) -> None:

        mse = (preds[:,:,:target.shape[2]] - target) ** 2

        mse = mse.mean(dim=(2, 3, 4))

        target_squared = target ** 2
        target_squared = target_squared.mean(dim=(2, 3, 4))

        if self.normalized:
            mse = mse / target_squared

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

class SimulationAE(SimulationMSE):

    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor) -> None:

        ae = torch.abs(preds[:, :, :target.shape[2]] - target)

        ae = ae.mean(dim=(2, 3, 4))

        self._update(ae, class_labels)

class SimulationRMSE3D(SimulationMSE):

    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor) -> None:

        mse = (preds[:,:,:target.shape[2]] - target) ** 2

        mse = mse.mean(dim=(2, 3, 4, 5))

        target_squared = target ** 2
        target_squared = target_squared.mean(dim=(2, 3, 4, 5))

        if self.normalized:
            mse = mse / target_squared

        mse = torch.sqrt(mse)

        self._update(mse, class_labels)

class SimulationMSE3D(SimulationMSE):

    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor) -> None:

        mse = (preds[:,:,:target.shape[2]] - target) ** 2

        mse = mse.mean(dim=(2, 3, 4, 5))

        self._update(mse, class_labels)

class SimulationMAE3D(SimulationMSE):

    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor) -> None:

        mse = (preds[:,:,:target.shape[2]] - target).abs()

        mse = mse.mean(dim=(2, 3, 4, 5))

        self._update(mse, class_labels)