import numpy as np
import random
import torch


def seeding(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)