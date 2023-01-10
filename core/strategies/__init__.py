from torchvision.models import ResNet
from torchvision.models.detection import FasterRCNN

from core.models.frcnn import fasterrcnn_embedding
from .base_sampling import *
from .coreset_sampling import *
from .k_means_sampling import *
from .least_confidence_sampling import *
from .random_sampling import *
from .representative_sampling import *
from ..models.resnet import resnet_embedding


def _embed(model, data, device='cpu'):
    if isinstance(model, FasterRCNN):
        return fasterrcnn_embedding(model, data, device)
    if isinstance(model, ResNet):
        return resnet_embedding(model, data, device)
    else:
        raise 'Unsupported model type for embedding.'


def active_sampling(
        labeled_data, unlabeled_data, number, limit=4096,
        strategy='representative', model=None, device='cpu'
):
    if 0 <= number < 1:
        number = len(unlabeled_data) * number
    number = int(number)

    _labeled_data = labeled_data
    np.random.shuffle(_labeled_data)
    _labeled_data = _labeled_data[:limit]

    _unlabeled_data = unlabeled_data
    np.random.shuffle(_unlabeled_data)
    _unlabeled_data = _unlabeled_data[:limit]

    _select_unlabeled_ids = []

    if strategy == 'representative':
        sampling = RepresentativeSampling()
        similar_score, select_ids = sampling.query(
            _embed(model, _labeled_data, device) \
                if len(_labeled_data) else None,
            _embed(model, _unlabeled_data, device)
        )
        _select_unlabeled_ids = select_ids

    elif strategy == 'coreset':
        sampling = CoresetSampling()
        _select_unlabeled_ids = sampling.query(
            _embed(model, _labeled_data, device) \
                if len(_labeled_data) else None,
            _embed(model, _unlabeled_data, device),
            min(number, len(_unlabeled_data))
        )

    elif strategy == 'kmeans':
        sampling = KMeansSampling()
        _select_unlabeled_ids = sampling.query(
            _embed(model, _unlabeled_data, device),
            min(number, len(_unlabeled_data))
        )

    elif strategy == 'lc':
        sampling = LeastConfidenceSampling()
        _select_unlabeled_ids = sampling.query(
            _embed(model, _unlabeled_data, device),
            min(number, len(_unlabeled_data))
        )

    elif strategy == 'random':
        sampling = RandomSampling()
        _select_unlabeled_ids = sampling.query(
            _unlabeled_data,
            min(number, len(_unlabeled_data))
        )

    else:
        raise f'the given strategy {strategy} is not supported.'

    _select_data = []
    for ids in _select_unlabeled_ids[:number]:
        _select_data.append(_unlabeled_data[ids])

    return _select_data
