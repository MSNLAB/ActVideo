import torchvision.transforms as T
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    def __init__(self, data, transforms=None):
        if transforms is None:
            transforms = T.Compose((
                T.ToTensor(),
                T.Resize(size=(128, 128)),
            ))
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, target = self.data[index]
        data = self.transforms(data)
        return data, target
