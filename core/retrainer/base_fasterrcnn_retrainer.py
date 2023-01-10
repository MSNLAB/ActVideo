import torch

from core.metrics.detection_metric import RetrainMetric, ValidMetric


class BaseFasterRCNNRetrainer:

    def __init__(self):
        self.tr_metric = RetrainMetric()
        self.val_metric = ValidMetric()

    def train_one_epoch(self, model, optim, data_loader, device,
                        epoch, num_epoch, scaler=None, *args, **kwargs):

        model.train()
        for images, targets in self.tr_metric.log_iter(epoch, num_epoch, data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optim)
                scaler.insert()
            else:
                losses.backward()
                optim.step()

            self.tr_metric.update(loss_dict, losses)
        return self.tr_metric.compute()

    def evaluate(self, model, data_loader, device):
        model.eval()
        for images, targets in self.val_metric.log_iter(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                outputs = model(images, targets)
            self.val_metric.update(outputs, targets)
        return self.val_metric.compute()
