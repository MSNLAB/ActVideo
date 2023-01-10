import numpy as np
import torch.nn.functional as F

from core.strategies import BaseSampling


class LeastConfidenceSampling(BaseSampling):

    def query(self, embeds, n):
        embeds = F.softmax(embeds, dim=1).numpy()
        _uncertainties = np.amax(embeds, axis=1)
        query_ids = _uncertainties.argsort()[:int(n)]
        return query_ids
