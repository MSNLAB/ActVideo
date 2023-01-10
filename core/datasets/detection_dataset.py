import torch
from PIL import Image
from torch.utils.data import Dataset

from core.datasets.detection_transforms import ToTensor, Compose, Resize


class DetectionDataset(Dataset):

    def __init__(self, frames, transforms=None):
        if transforms is None:
            transforms = Compose((
                ToTensor(),
                Resize(size=(720, 1280)),
            ))
        self.frames = frames
        self.transforms = transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame_info = self.frames[index]
        img = Image.open(frame_info['path']).convert("RGB")
        target = {"boxes": torch.as_tensor(frame_info['boxes'], dtype=torch.float32),
                  "labels": torch.as_tensor(frame_info['labels'], dtype=torch.int64),}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
