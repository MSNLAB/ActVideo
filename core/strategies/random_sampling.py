import numpy as np

from core.strategies import BaseSampling


class RandomSampling(BaseSampling):

    def query(self, unlabeled_data, number, replace=False):
        ids = np.random.choice(list(range(len(unlabeled_data))), number, replace)
        return ids.tolist()
