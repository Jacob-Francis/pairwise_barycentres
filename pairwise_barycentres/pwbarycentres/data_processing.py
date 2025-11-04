import torch
import numpy as np
from torch_numpy_processing import TorchNumpyProcess

class BarycentreProcessor:
    def __init__(self, data):
        self.data = data
        self.processor = TorchNumpyProcess()

    def compute_barycentre(self):
        tensor_data = self.processor.to_tensor(self.data)
        barycentre = torch.mean(tensor_data, dim=0)
        return self.processor.to_numpy(barycentre)