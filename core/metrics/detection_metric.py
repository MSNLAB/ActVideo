import time

from numpy import average as avg
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm


class RetrainMetric:

    def __init__(self):
        self.metrics = {}

    def reset_metrics(self):
        self.metrics = {
            'loss_classifier': [], 'loss_box_reg': [],
            'loss_objectness': [], 'loss_rpn_box_reg': [],
            'total_loss': [],
        }

    def update(self, loss_dict, total_loss):
        self.metrics['loss_classifier'].append(loss_dict['loss_classifier'].detach().cpu().item())
        self.metrics['loss_box_reg'].append(loss_dict['loss_box_reg'].detach().cpu().item())
        self.metrics['loss_objectness'].append(loss_dict['loss_objectness'].detach().cpu().item())
        self.metrics['loss_rpn_box_reg'].append(loss_dict['loss_rpn_box_reg'].detach().cpu().item())
        self.metrics['total_loss'].append(total_loss.detach().cpu().item())

    def compute(self):
        return {
            'loss_classifier': avg(self.metrics['loss_classifier']),
            'loss_box_reg': avg(self.metrics['loss_box_reg']),
            'loss_objectness': avg(self.metrics['loss_objectness']),
            'loss_rpn_box_reg': avg(self.metrics['loss_rpn_box_reg']),
            'total_loss': avg(self.metrics['total_loss']),
        }

    def log_iter(self, epoch, num_epoch, data_loader):
        loop = tqdm(enumerate(data_loader, 1), total=len(data_loader),
                    desc=f'Epoch [{epoch}/{num_epoch}]')

        self.reset_metrics()
        data_load_time = []
        train_process_time = []

        end_time = time.time()
        for idx, (images, targets) in loop:
            start_time = time.time()
            data_load_time.append(start_time - end_time)

            yield images, targets

            end_time = time.time()
            train_process_time.append(end_time - start_time)

            loop.set_postfix(
                loss=f"{avg(self.metrics['total_loss']):.2f}",
                data=f"{avg(data_load_time):.3f}s/it",
                train=f"{avg(train_process_time):.3f}s/it",
            )


class ValidMetric:

    def __init__(self):
        self.detection_metric = MeanAveragePrecision()

    def reset_metrics(self):
        self.detection_metric = MeanAveragePrecision()

    def update(self, pred_ans, target):
        pred_ans = [{k: v.detach().cpu() for k, v in out.items()} for out in pred_ans]
        target = [{k: v.detach().cpu() for k, v in t.items()} for t in target]
        self.detection_metric.update(pred_ans, target)

    def compute(self):
        detection_ans = {k: v.cpu().detach().item() \
                         for k, v in self.detection_metric.compute().items()}
        return detection_ans

    def log_iter(self, data_loader):
        loop = tqdm(enumerate(data_loader, 1), total=len(data_loader), desc=f'Test')

        self.reset_metrics()
        data_load_time = []
        pred_process_time = []

        end_time = time.time()
        for idx, (images, targets) in loop:
            start_time = time.time()
            data_load_time.append(start_time - end_time)

            yield images, targets

            end_time = time.time()
            pred_process_time.append(end_time - start_time)

            loop.set_postfix(
                data=f"{avg(data_load_time):.3f}s/it",
                pred=f"{avg(pred_process_time):.3f}s/it",
            )

        ans = self.compute()
        print(
            f"IoU metric: bbox; ",
            f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ans['map']:.3f}",
            f"Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {ans['map_50']:.3f}",
            f"Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {ans['map_75']:.3f}",
            f"Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {ans['map_small']:.3f}",
            f"Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {ans['map_medium']:.3f}",
            f"Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {ans['map_large']:.3f}",
            f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {ans['mar_1']:.3f}",
            f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {ans['mar_10']:.3f}",
            f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ans['mar_100']:.3f}",
            f"Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {ans['mar_small']:.3f}",
            f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {ans['mar_medium']:.3f}",
            f"Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {ans['mar_large']:.3f}",
            sep='\n',
        )
