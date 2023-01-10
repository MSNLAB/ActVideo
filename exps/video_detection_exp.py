import argparse
import json
import os
from datetime import datetime

import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from exps.common import prepare_model, prepare_dataset
from core.strategies import active_sampling
from core.datasets import DetectionDataset
from core.examplers import RandomExampler
from core.retrainer import RehearsalFasterRCNNRetrainer
from core.utils.misc import same_seeds


def _collate_fn(batch):
    return tuple(zip(*batch))


def run_exp(dataset_name, dataset_root, sequences, model_name, num_classes,
            pretrained_weights, al_strategy, al_ratio, cache_max, epoch_num,
            lr, weight_decay, batch_size, num_workers, seed, device):
    same_seeds(seed)
    video_datasets = prepare_dataset(dataset_name, dataset_root, sequences)

    print(f"Video datastream loading done:",
          f"  - dataset: {dataset_name};",
          f"  - sequences: {sequences};",
          f"  - root: {dataset_root};",
          f"  - batch_size: {batch_size}; num_workers: {num_workers}.",
          sep='\n')

    model = prepare_model(model_name, num_classes, pretrained_weights).to(device)
    optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"Model loading done:",
          f"  - model: {model_name};",
          f"  - num_classes: {num_classes};",
          f"  - device: {device};",
          f"  - weights: {pretrained_weights}.",
          sep='\n')

    cache = RandomExampler(maxsize=cache_max)
    engine = RehearsalFasterRCNNRetrainer()

    exp_ans = {'retrain': {}, 'test': {}}

    for seq_name, dataset in video_datasets.items():
        exp_ans['retrain'][f'{seq_name}'] = []
        exp_ans['test'][f'{seq_name}-before'] = []
        exp_ans['test'][f'{seq_name}-after'] = []

        _frames = dataset.frames
        np.random.shuffle(_frames)
        tr_frames = _frames[:int(0.7 * len(_frames))]
        val_frames = _frames[int(0.7 * len(_frames)):]

        # before evaluation
        val_dataset = DetectionDataset(val_frames)
        for images, targets in val_dataset:
            batch = (([images], [targets]),)
            eval_ans = engine.evaluate(model, batch, device)
            exp_ans['test'][f'{seq_name}-before'].append(eval_ans)

        # active continual learning for drifted data
        select_frames = []
        for epoch in range(1, epoch_num + 1):
            select_frames.extend(active_sampling(
                labeled_data=[*select_frames, *cache.caches],
                unlabeled_data=tr_frames, number=al_ratio,
                strategy=al_strategy, model=model,
                device=device)
            )

            memory_data = DetectionDataset(cache.caches)
            drift_data = DataLoader(
                DetectionDataset(select_frames),
                collate_fn=_collate_fn,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

            tr_ans = engine.train_one_epoch(
                model, optim, drift_data, memory_data,
                device, epoch, epoch_num)

            exp_ans['retrain'][f'{seq_name}'].append(tr_ans)

        # after evaluation
        val_dataset = DetectionDataset(val_frames)
        for images, targets in val_dataset:
            batch = (([images], [targets]),)
            eval_ans = engine.evaluate(model, batch, device)
            exp_ans['test'][f'{seq_name}-after'].append(eval_ans)

        # rehearsal memory update
        update_size = min(len(select_frames) // epoch_num, cache.used_size)
        if not cache.insertable(update_size):
            cache.random_reduce(update_size)
        cache.random_insert(select_frames, update_size)

    return exp_ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--sequences', type=str, nargs='+', required=True)

    parser.add_argument('--model', type=str, required=False, default='fasterrcnn_mobilenet_v3_large_fpn')
    parser.add_argument('--num_classes', type=int, required=False, default=12)
    parser.add_argument('--pretrained_weights', type=str, required=False, default=None)

    parser.add_argument('--al_strategy', type=str, required=False, default='random')
    parser.add_argument('--al_ratio', type=float, required=False, default=0.1)
    parser.add_argument('--cache_max', type=int, required=False, default=100)

    parser.add_argument('--epoch_num', type=int, required=False, default=10)
    parser.add_argument('--lr', type=float, required=False, default=1e-3)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-5)

    parser.add_argument('--result_dir', type=str, required=False, default='./')
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--num_workers', type=int, required=False, default=0)
    parser.add_argument('--seed', type=int, required=False, default=42069)
    parser.add_argument('--device', type=str, required=False, default='cuda')

    args = vars(parser.parse_args())
    exp_ans = run_exp(
        args['dataset_name'], args['dataset_root'], args['sequences'],
        args['model'], args['num_classes'], args['pretrained_weights'],
        args['al_strategy'], args['al_ratio'], args['cache_max'],
        args['epoch_num'], args['lr'], args['weight_decay'], args['batch_size'],
        args['num_workers'], args['seed'], args['device'])

    ans = {'setting': args, **exp_ans}
    print(ans)

    save_path = os.path.join(
        args['result_dir'], f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    with open(f'{save_path}.json', "w") as f:
        json.dump(ans, f, indent=2)
