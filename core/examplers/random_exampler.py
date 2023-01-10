import numpy as np

from .base_exampler import BaseExampler


class RandomExampler(BaseExampler):

    def random_replay(self, number=None, replace=False):
        if number is None:
            number = self.used_size
        elif 0 <= number < 1:
            number = self.used_size * number
        number = int(number)

        idx = np.random.choice(list(range(self.used_size)), number, replace)
        return super(RandomExampler, self).replay(idx.tolist())

    def random_insert(self, data, number=None):
        if not isinstance(data, (list, tuple)):
            data = (data,)

        if number is None:
            number = len(data)
        elif 0 <= number < 1:
            number = len(data) * number
        number = int(number)

        _data = np.random.choice(data, number, replace=False)
        return super(RandomExampler, self).insert(_data.tolist())

    def random_reduce(self, number):
        if 0 <= number < 1:
            number = self.used_size * number
        number = int(number)

        rand_idx = np.random.choice(
            list(range(self.used_size)), number, replace=False)
        return super(RandomExampler, self).reduce(rand_idx.tolist())
