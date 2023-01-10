import time

import torch
import torch.nn.functional as F
from numpy import average as avg
from tqdm import tqdm


class RetrainMetric:

    def __init__(self):
        self.metrics = {}

    def reset_metrics(self):
        self.metrics = {
            'loss_distill': [],
            'loss_clf': [],
            'total_loss': [],
            'tr_cnt': [],
        }

    def update(self, loss_dict, total_loss, tr_cnt):
        self.metrics['loss_distill'].append(loss_dict['loss_distill'].detach().cpu().item())
        self.metrics['loss_clf'].append(loss_dict['loss_clf'].detach().cpu().item())
        self.metrics['total_loss'].append(total_loss.detach().cpu().item())
        self.metrics['tr_cnt'].append(tr_cnt)

    def compute(self):
        return {
            'loss_distill': avg(self.metrics['loss_distill']),
            'loss_clf': avg(self.metrics['loss_clf']),
            'total_loss': avg(self.metrics['total_loss']),
            'tr_cnt': sum(self.metrics['tr_cnt']),
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
        self.classification_loss = []
        self.classification_top_1 = []
        self.classification_top_5 = []

    def reset_metrics(self):
        self.classification_loss = []
        self.classification_top_1 = []
        self.classification_top_5 = []

    def update(self, pred, target):
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        self.classification_loss.append(F.cross_entropy(pred, target).detach().item())
        self.classification_top_1.append(
            torch.eq(pred.argmax(dim=1), target).sum().float()
            .cpu().detach().item() / len(target)
        )
        self.classification_top_5.append(
            torch.eq(pred.topk(5, dim=-1, largest=True, sorted=True)[1], target.view(-1, 1))
            .sum().float().cpu().detach().item() / len(target)
        )

    def compute(self):
        return {
            'loss': avg(self.classification_loss),
            'top-1': avg(self.classification_top_1),
            'top-5': avg(self.classification_top_5),
        }

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
                loss=f"{avg(self.classification_loss):.2f}",
                top_1=f"{avg(self.classification_top_1):.2f}",
                top_5=f"{avg(self.classification_top_5):.2f}",
                data=f"{avg(data_load_time):.3f}s/it",
                pred=f"{avg(pred_process_time):.3f}s/it",
            )
