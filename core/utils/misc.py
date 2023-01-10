import random

import numpy as np
import torch


def to_device(*data, device):
    if len(data) == 1:
        data = data[0]

    ans = []
    for _data in data:
        if isinstance(_data, (tuple, list)):
            ans.append(to_device(_data, device=device))
        elif isinstance(_data, dict):
            ans.append({_k: to_device(_v, device=device) \
                        for _k, _v in _data.items()})
        else:
            ans.append(_data.to(device))
    return ans if len(data) > 1 else ans[0]


def same_seeds(seed=42069):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
