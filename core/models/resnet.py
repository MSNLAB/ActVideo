import torch
from torchvision.models import resnet101
from torchvision.models import resnet152
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50

from core.datasets.classification_dataset import ClassificationDataset

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
]


def resnet_embedding(model, data, device):
    dataset = ClassificationDataset(data)

    embeds = []
    model.eval()
    for data, _ in dataset:
        data = data.unsqueeze(0).to(device)
        with torch.no_grad():
            # embed = model.conv1(data)
            # embed = model.bn1(embed)
            # embed = model.relu(embed)
            # embed = model.maxpool(embed)
            #
            # embed = model.layer1(embed)
            # embed = model.layer2(embed)
            # embed = model.layer3(embed)
            # embed = model.layer4(embed)
            #
            # embed = model.avgpool(embed)
            # embed = torch.flatten(embed, 1)
            embed = model.forward(data)

        embeds.append(embed.view(1, -1).cpu().detach())

    embeds = torch.cat(embeds, dim=0)
    return embeds
