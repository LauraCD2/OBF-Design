from typing import Union
import numpy as np 
import torch

def r2_fn(y, y_pred):
    ss_res = torch.sum(torch.square(y - y_pred))
    ss_tot = torch.sum(torch.square(y - torch.mean(y)))
    return 1 - ss_res / ss_tot

mae_fn  = lambda y, y_pred: torch.abs(y - y_pred).mean()
mape_fn = lambda y, y_pred: torch.abs((y - y_pred) / y).mean()
mse_fn  = lambda y, y_pred: torch.square(y - y_pred).mean()

class AverageMeter(object):
    """Stores values and keeps track of averages and standard deviations.

    :param str name: meter name for printing
    :param str fmt: meter format for printing
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset meter values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.std = 0.0
        self.sum2 = 0.0

    def update(self, val: Union[np.ndarray, float, int], n: int = 1) -> None:
        """Update average meter.

        :param numpy.ndarray, float, int val: either array (i.e. batch) of values or single value
        :param int n: weight, defaults to 1
        """
        if isinstance(val, np.ndarray):
            self.val = np.mean(val)
            self.sum += np.sum(val) * n
            self.sum2 += np.sum(val**2) * n
            self.count += n * np.prod(val.shape)
        else:
            self.val = val
            self.sum += val * n
            self.sum2 += val**2 * n
            self.count += n

        self.avg = self.sum / self.count
        var = self.sum2 / self.count - self.avg**2
        self.std = np.sqrt(var) if var > 0 else 0

    def __str__(self):
        fmtstr = "{name}={avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)



def print_results(classifier_name, dataset_name, dict_metrics):
    print('#================================================#')
    print(f'Classifier: {classifier_name}, Dataset: {dataset_name}')
    for name, metrics in dict_metrics.items():
        print(f'{name} -> MSE: {metrics["MSE"]:.4f}, MAE: {metrics["MAE"]:.4f}, MAPE: {metrics["MAPE"]:.4f} R2: {metrics["R2"]:.4f}')
    print('#================================================#') 
