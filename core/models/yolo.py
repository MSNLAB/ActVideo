import importlib.util
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

__all__ = (
    'yolov5_nano',
    'yolov5_small',
    'yolov5_medium',
    'yolov5_large',
    'yolov5_xlarge',
    'yolov5_nano_p6',
    'yolov5_small_p6',
    'yolov5_medium_p6',
    'yolov5_large_p6',
    'yolov5_xlarge_p6',
)


def format_targets_as_yolo(inputs, targets):
    _format_targets = []
    kwargs_factory = {'device': inputs.device, 'dtype': torch.float32}
    for batch_idx, target in enumerate(targets):
        c, h, w = inputs[batch_idx].shape
        for bbox_idx in range(len(target['labels'])):
            batch_idx = torch.tensor([batch_idx], **kwargs_factory)
            cls = torch.tensor((target['labels'][bbox_idx],), **kwargs_factory)
            bbox = torch.tensor(target['boxes'][bbox_idx], **kwargs_factory)
            bbox[[0, 2]] /= w
            bbox[[1, 3]] /= h
            _format_targets.append(torch.cat((batch_idx, cls, bbox), dim=0).unsqueeze(0))
    return torch.cat(_format_targets, dim=0)


def scale_bbox(bbox, shape_0, shape_1):
    scale_h = shape_1[0] / shape_0[0]
    scale_w = shape_1[1] / shape_0[1]
    bbox[:, [0, 2]] *= scale_w
    bbox[:, [1, 3]] *= scale_h
    return bbox


class Yolo(nn.Module):
    __yolo_v5_repository__ = 'ultralytics/yolov5:v6.2'

    def __init__(self, *, yolo_cfg, pretrained=True, channels=3, size=(640, 640), classes=80,
                 nms_thresh=0.25, iou_thresh=0.45, agnostic=False, max_det=300,
                 autoshape=False, _verbose=False) -> None:
        super(Yolo, self).__init__()

        self.yolo_cfg = yolo_cfg
        self.pretrained = pretrained
        self.channels = channels
        self.classes = classes
        self.size = size
        self.nms_thresh = nms_thresh
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic
        self.max_det = max_det
        self.autoshape = autoshape
        self._verbose = _verbose

        # load official yolov5 backbone
        self.detector = torch.hub.load(self.__yolo_v5_repository__, yolo_cfg, pretrained,
                                       channels, classes, autoshape, _verbose)
        if self.detector.__class__.__name__ in ('DetectMultiBackend', 'Autoshape'):
            self.detector = self.detector.model
            self.detector = self.detector.cpu()

        # load official yolov5 nms function
        assert importlib.util.find_spec('utils.general'), \
            f"could not import Yolov5 module: utils.general"
        self.non_max_suppression = importlib.import_module('utils.general').non_max_suppression

        # load official yolov5 loss function
        assert importlib.util.find_spec('utils.loss')
        self.loss_fn = importlib.import_module('utils.loss').ComputeLoss(self.detector)

    def forward(self, inputs, targets=None):
        if isinstance(inputs, list):
            inputs = torch.cat([x.unsqueeze(0) for x in inputs])

        if not self.training:
            return self.inference(inputs)

        assert targets, "targets could not be None in the training stage"

        bs, c, h, w = inputs.shape
        preds = self.detector(inputs)
        _format_targets = format_targets_as_yolo(inputs, targets)
        _, loss_items = self.loss_fn(preds, _format_targets)

        loss_dict = {
            'loss_box': bs * loss_items[0],
            'loss_obj': bs * loss_items[1],
            'loss_cls': bs * loss_items[2],
        }

        return loss_dict

    def inference(self, inputs):
        bs, c, h, w = inputs.shape

        inputs = F.resize(inputs, self.size)
        preds, _ = self.detector(inputs)

        preds = self.non_max_suppression(
            prediction=preds,
            conf_thres=self.nms_thresh,
            iou_thres=self.iou_thresh,
            agnostic=self.agnostic,
            max_det=self.max_det
        )
        preds = [scale_bbox(pred, self.size, (h, w)) for pred in preds]

        results = []
        for batch_idx, pred in enumerate(preds):
            results.append({
                "boxes": pred[:, :4],
                "scores": pred[:, 4],
                "labels": pred[:, -1],
            })

        return results


def yolov5_nano(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5n', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_small(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5s', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_medium(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5m', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_large(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5l', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_xlarge(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5x', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_nano_p6(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5n6', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_small_p6(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5s6', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_medium_p6(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5m6', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_large_p6(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5l6', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)


def yolov5_xlarge_p6(*, pretrained=True, channels=3, classes=80, **kwargs):
    return Yolo(yolo_cfg='yolov5x6', pretrained=pretrained,
                channels=channels, classes=classes, **kwargs)
