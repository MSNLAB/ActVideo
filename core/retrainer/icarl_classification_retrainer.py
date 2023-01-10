import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from core.metrics.classification_metric import RetrainMetric, ValidMetric


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(target.device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class iCaRLClassificationRetrainer:

    def __init__(self):
        self.tr_metric = RetrainMetric()
        self.val_metric = ValidMetric()

    def train_one_epoch(self, model, prev_model, optim, incre_loader,
                        rehearsal_dataset, device, epoch, num_epoch, scaler=None):
        concat_loader = incre_loader
        if rehearsal_dataset is not None and len(rehearsal_dataset):
            concat_dataset = ConcatDataset([incre_loader.dataset, rehearsal_dataset])
            concat_loader = DataLoader(
                concat_dataset,
                batch_size=incre_loader.batch_size, shuffle=True,
                num_workers=incre_loader.num_workers, collate_fn=incre_loader.collate_fn,
                pin_memory=incre_loader.pin_memory, drop_last=incre_loader.drop_last,
                timeout=incre_loader.timeout, worker_init_fn=incre_loader.worker_init_fn,
                multiprocessing_context=incre_loader.multiprocessing_context,
                persistent_workers=incre_loader.persistent_workers,
            )

        model.train(), prev_model.eval()
        for inputs, targets in self.tr_metric.log_iter(epoch, num_epoch, concat_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                prev_preds = prev_model(inputs)
                preds = model(inputs)

                loss_clf = F.binary_cross_entropy_with_logits(
                    input=preds,
                    target=get_one_hot(targets, preds.shape[-1]).to(device)
                )

                if rehearsal_dataset is not None and len(rehearsal_dataset):
                    loss_distill = F.binary_cross_entropy_with_logits(
                        input=preds[:, :prev_preds.shape[-1]],
                        target=torch.sigmoid(prev_preds[:, :prev_preds.shape[-1]]).to(device)
                    )
                else:
                    loss_distill = torch.zeros((1,)).to(loss_clf.device)

                total_loss = loss_clf + loss_distill

                loss_dict = {
                    'loss_distill': loss_distill,
                    'loss_clf': loss_clf,
                    'total_loss': total_loss,
                }

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optim)
                scaler.insert()
            else:
                total_loss.backward()
                optim.step()

            self.tr_metric.update(loss_dict, total_loss, len(inputs))
        return self.tr_metric.compute()

    def evaluate(self, model, data_loader, device):
        model.eval()
        for inputs, targets in self.val_metric.log_iter(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                preds = model(inputs)
            self.val_metric.update(preds, targets)
        return self.val_metric.compute()
