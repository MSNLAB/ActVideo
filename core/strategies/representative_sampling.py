import torch
import torch.nn.functional as F

from core.strategies.base_sampling import BaseSampling


class RepresentativeSampling(BaseSampling):

    def __init__(self, lambda_score=1.0):
        self.lambda_score = lambda_score

    def query(self, labeled_embeds=None, unlabeled_embeds=None):
        if labeled_embeds is None:
            labeled_embeds = torch.zeros_like(unlabeled_embeds) \
                .to(unlabeled_embeds.device)
        
        label_mean = torch.mean(labeled_embeds, dim=0)
        unlabeled_mean = torch.mean(unlabeled_embeds, dim=0)

        label_score = F.cosine_similarity(label_mean, unlabeled_embeds, dim=1)
        unlabeled_score = F.cosine_similarity(unlabeled_mean, unlabeled_embeds, dim=1)

        representativeness = self.lambda_score * label_score - unlabeled_score
        ranks = torch.sort(representativeness, descending=False)
        scores, ids = ranks

        return scores, ids
