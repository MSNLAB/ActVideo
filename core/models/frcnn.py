import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

from core.datasets import DetectionDataset

__all__ = [
    'fasterrcnn_resnet50_fpn',
    'fasterrcnn_mobilenet_v3_large_fpn',
    'fasterrcnn_mobilenet_v3_large_320_fpn'
]


def fasterrcnn_embedding(model, frame_infos, device='cpu'):
    dataset = DetectionDataset(frame_infos)
    embedding_model = model.backbone

    embeds = []
    embedding_model.eval()
    for data, _ in dataset:
        data = data.to(device)
        with torch.no_grad():
            embed = embedding_model(data)['pool']
        embeds.append(embed.view(1, -1).detach().cpu())

    embeds = torch.cat(embeds, dim=0)
    return embeds
