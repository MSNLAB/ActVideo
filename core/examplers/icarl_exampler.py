import math
import numpy as np
import torch
from torch import nn

from core.examplers import BaseExampler


def add_n_classes(classifier: nn.Linear, n):
    require_bias = classifier.bias is not None

    weight = classifier.weight.data
    bias = classifier.bias.data if require_bias else None

    _classifier = nn.Linear(
        classifier.in_features,
        classifier.out_features + n,
        require_bias
    ).to(classifier.device)

    _classifier.weight.data[:classifier.out_features] = weight
    if require_bias:
        _classifier.bias.data[:classifier.out_features] = bias

    return _classifier


class iCaRLExampler(BaseExampler):

    def __init__(self, maxsize=4096):
        super(iCaRLExampler, self).__init__(maxsize)

    @property
    def used_size(self):
        return sum([len(e) for e in self.caches if e is not None and len(e)])

    @property
    def n_classes(self):
        cnt = len([None for e in self.caches if e is not None and len(e)])
        return cnt if cnt > 0 else 1

    @property
    def m(self):
        return math.ceil(self.maxsize / self.n_classes)

    def insert(self, model, dataloader, device):
        imgs, ids, features = [], [], []

        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                imgs.append(x.clone().detach().cpu())
                ids.append(y.clone().detach().cpu())
                features.append(model.forward(x).clone().detach().cpu())

        imgs = torch.cat(imgs).detach().numpy()
        ids = torch.cat(ids).detach().numpy()
        features = torch.cat(features).detach().numpy()

        for idx in np.unique(ids):
            _ids = np.argwhere(ids == idx).squeeze(axis=1)

            _imgs = imgs[_ids]
            _features = features[_ids]
            _mean = np.mean(_features)

            examplars = []
            examplars_fea = []

            for i in range(self.m):
                p = _mean - (_features + np.sum(examplars_fea, axis=0)) / (i + 1)
                p = np.linalg.norm(p, axis=1)
                min_idx = np.argmin(p)
                examplars.append((_imgs[min_idx].transpose(1, 2, 0), idx))
                examplars_fea.append(_features[min_idx])

            while len(self.caches) - 1 < idx:
                self.caches.append([])
            self.caches[idx] = examplars

    def reduce(self):
        m = self.m
        for idx, e in enumerate(self.caches):
            if e is not None and len(e):
                self.caches[idx] = self.caches[idx][:m]

    def replay(self):
        return sum([e for e in self.caches], [])
