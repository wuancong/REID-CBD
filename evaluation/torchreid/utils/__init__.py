from __future__ import absolute_import
import torch
from .tools import *
from .rerank import re_ranking
from .loggers import *
from .avgmeter import *
from .torchtools import *
from .model_complexity import compute_model_complexity
from .comm import *


def parse_parameters(model, keywords):
    parameters = []
    for name, param in model.named_parameters():
        if keywords in name:
            parameters.append(param)
    return parameters


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray