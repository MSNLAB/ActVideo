import os

import pandas as pd
from torch.utils.data import DataLoader

from core.datasets.detection_dataset import DetectionDataset

annotation_cols = ('frame_index', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width',
                   'bbox_height', 'score', 'object_category', 'truncation', 'occlusion',)

object_category = ('ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van',
                   'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others',)


def _collate_fn(batch):
    return tuple(zip(*batch))


def collect_frames(root, sequences):
    if not isinstance(sequences, (list, tuple)):
        sequences = (sequences,)

    frames = []
    for sequence in sequences:
        frame_path = os.path.join(root, 'sequences', sequence)
        frame_names = list(os.listdir(frame_path))

        annotation_path = os.path.join(root, 'annotations', sequence + '.txt')
        annotations = pd.read_csv(annotation_path, header=None, names=annotation_cols)

        for frame_name in frame_names:
            _id = int(frame_name.split('.')[0])
            _path = os.path.join(frame_path, frame_name)
            _labels = annotations[annotations['frame_index'] == _id]

            boxes = []
            labels = []
            for _idx, _label in _labels.iterrows():
                label = _label['object_category']
                if label != 0:
                    x_min = int(_label['bbox_left'])
                    y_min = int(_label['bbox_top'])
                    x_max = x_min + int(_label['bbox_width'])
                    y_max = y_min + int(_label['bbox_height'])
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)

            if len(boxes) != 0 and len(labels) != 0:
                frames.append({'path': _path, 'sequence': sequence, 'frame_index': _id,
                               'boxes': boxes, 'labels': labels})

    return frames


def visdrone2019_dataloader(root=None, sequences=None, frames=None, transforms=None, **kwargs):
    kwargs_factory = {'collate_fn': _collate_fn, **kwargs}
    return DataLoader(VisDrone2019(root, sequences, frames, transforms), **kwargs_factory)


class VisDrone2019(DetectionDataset):

    def __init__(self, root=None, sequences=None, frames=None, transforms=None):
        if not isinstance(sequences, (list, tuple)):
            sequences = (sequences,)

        self.root = root
        self.sequences = sequences

        if frames is None:
            frames = collect_frames(root, sequences)
        super(VisDrone2019, self).__init__(frames, transforms)
