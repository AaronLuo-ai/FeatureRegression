import torch
import numpy as np
import torch.nn as nn


def load_features(path):
    model = torch.load(path)
    return model