from torch.utils.data import ConcatDataset, DataLoader

from core.retrainer import BaseFasterRCNNRetrainer


class RehearsalFasterRCNNRetrainer(BaseFasterRCNNRetrainer):

    def train_one_epoch(self, model, optim, data_loader, rehearsal_dataset,
                        device, epoch, num_epoch, scaler=None, *args, **kwargs):
        concat_loader = data_loader
        if rehearsal_dataset is not None:
            concat_dataset = ConcatDataset([data_loader.dataset, rehearsal_dataset])
            concat_loader = DataLoader(
                concat_dataset,
                batch_size=data_loader.batch_size, shuffle=True,
                num_workers=data_loader.num_workers, collate_fn=data_loader.collate_fn,
                pin_memory=data_loader.pin_memory, drop_last=data_loader.drop_last,
                timeout=data_loader.timeout, worker_init_fn=data_loader.worker_init_fn,
                multiprocessing_context=data_loader.multiprocessing_context,
                persistent_workers=data_loader.persistent_workers,
            )

        return super().train_one_epoch(model, optim, concat_loader, device,
                                       epoch, num_epoch, scaler, *args, **kwargs)
