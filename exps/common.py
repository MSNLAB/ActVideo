import numpy as np
import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from core.datasets.classification_dataset import ClassificationDataset
from core.models import fasterrcnn_mobilenet_v3_large_fpn
from core.models import resnet18
from videos.core50 import CORE50
from videos.dashcam import DashCamDataset
from videos.traffic import TrafficDataset
from videos.visdrone import VisDrone2019


def prepare_model(model, num_classes=92, weights=None):
    if model == 'fasterrcnn_mobilenet_v3_large_fpn':
        model = fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=True, trainable_backbone_layers=3)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes=num_classes)
    elif model == 'resnet18':
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if weights is not None:
        model.load_state_dict(torch.load(weights))

    return model


def prepare_dataset(dataset_name, dataset_root, sequences=None):
    if dataset_name == 'visdrone':
        return _prepare_dataset_for_visdrone(dataset_root, sequences)
    elif dataset_name == 'dashcam':
        return _prepare_dataset_for_dashcam(dataset_root, sequences)
    elif dataset_name == 'traffic':
        return _prepare_dataset_for_traffic(dataset_root, sequences)
    elif dataset_name == 'core50':
        return _prepare_dataset_for_core50(dataset_root)
    else:
        raise f'the given dataset name {dataset_name} is not supported.'


def _prepare_dataset_for_core50(dataset_root, cumul_step=8, category_level=10):
    core50 = CORE50(root=dataset_root, preload=True, scenario='nic', cumul=False, run=0)
    batches = [(x, y) for x, y in core50]
    datasets = {}
    for idx in range(len(batches) // cumul_step):
        x = np.concatenate(
            [batch[0] for batch in batches[idx * cumul_step:(idx + 1) * cumul_step]], axis=0)
        y = np.concatenate(
            [batch[1] for batch in batches[idx * cumul_step:(idx + 1) * cumul_step]], axis=0)
        datasets[idx] = ClassificationDataset([(
            data.astype(np.float32) / 255,
            target.astype(np.long) // (5 if category_level == 10 else 1),
        ) for data, target in zip(x, y)])
    return datasets


def _prepare_dataset_for_visdrone(dataset_root, sequences):
    datasets = {seq: VisDrone2019(dataset_root, seq) for seq in sequences}
    return datasets


def _prepare_dataset_for_dashcam(dataset_root, sequences):
    datasets = {seq: DashCamDataset(dataset_root, seq) for seq in sequences}
    return datasets


def _prepare_dataset_for_traffic(dataset_root, sequences):
    datasets = {seq: TrafficDataset(dataset_root, seq) for seq in sequences}
    return datasets
