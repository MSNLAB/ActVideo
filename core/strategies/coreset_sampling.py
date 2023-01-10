import numpy as np
import torch
from scipy.spatial import distance_matrix

from core.strategies.base_sampling import BaseSampling


class CoresetSampling(BaseSampling):

    def query(self, labeled_embeds, unlabeled_embeds, number):
        if labeled_embeds is None:
            labeled_embeds = torch.zeros_like(unlabeled_embeds) \
                .to(unlabeled_embeds.device)

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled_embeds[0, :].reshape((1, labeled_embeds.shape[1])), unlabeled_embeds),
                          axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled_embeds.shape[0], 100):
            if j + 100 < labeled_embeds.shape[0]:
                dist = distance_matrix(labeled_embeds[j:j + 100, :], unlabeled_embeds)
            else:
                dist = distance_matrix(labeled_embeds[j:, :], unlabeled_embeds)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(number - 1):
            dist = distance_matrix(unlabeled_embeds[greedy_indices[-1], :].reshape((1, unlabeled_embeds.shape[1])),
                                   unlabeled_embeds)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return greedy_indices
