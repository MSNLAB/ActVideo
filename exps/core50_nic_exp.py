import argparse
import copy
import json
import os
from datetime import datetime

import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from exps.common import prepare_model, _prepare_dataset_for_core50
from core.strategies import active_sampling
from core.datasets.classification_dataset import ClassificationDataset
from core.examplers.icarl_exampler import iCaRLExampler
from core.retrainer.icarl_classification_retrainer import iCaRLClassificationRetrainer
from core.utils.misc import same_seeds


def _prepare_drift_pool(drifted_data, pre_data=None):
    _data = drifted_data
    np.random.shuffle(_data)
    tr_data = _data[:int(0.8 * len(_data))]
    val_data = _data[int(0.2 * len(_data)):]

    drift_pool = tr_data[:int(0.2 * len(tr_data))]
    if pre_data is not None:
        drift_pool.extend(pre_data[:int(0.8 * len(tr_data))])
    np.random.shuffle(drift_pool)

    return drift_pool, tr_data, val_data


def run_exp(core50_root, core50_cumul, core50_category, model_name, num_classes,
            pretrained_weights, al_strategy, al_ratio, cache_max, epoch_num,
            lr, weight_decay, batch_size, num_workers, seed, device):
    same_seeds(seed)
    video_datasets = _prepare_dataset_for_core50(
        core50_root, core50_cumul, core50_category)

    print(f"CORe50 datastream loading done:",
          f"  - root: {core50_root};",
          f"  - batch_size: {batch_size}; ",
          f"  - num_workers: {num_workers}.",
          sep='\n')

    model = prepare_model(model_name, num_classes, pretrained_weights).to(device)

    print(f"Model loading done:",
          f"  - model: {model_name};",
          f"  - num_classes: {num_classes};",
          f"  - device: {device};",
          f"  - weights: {pretrained_weights}.",
          sep='\n')

    cache = iCaRLExampler(maxsize=cache_max)
    engine = iCaRLClassificationRetrainer()

    exp_ans = {'retrain': {}, 'test': {}}

    pre_data = None
    for seq_name, dataset in video_datasets.items():
        exp_ans['retrain'][f'{seq_name}'] = []
        exp_ans['test'][f'{seq_name}-before'] = []
        exp_ans['test'][f'{seq_name}-after'] = []

        # simulate drifted pool
        drift_pool, tr_data, val_data = _prepare_drift_pool(dataset.data, pre_data)
        pre_data = tr_data

        val_loader = DataLoader(
            ClassificationDataset(val_data),
            batch_size=batch_size,
            num_workers=num_workers
        )

        # before evaluation
        eval_ans = engine.evaluate(model, val_loader, device)
        exp_ans['test'][f'{seq_name}-before'].append(eval_ans)

        # active continual learning for drifted data
        _prev_model = copy.deepcopy(model)
        optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        select_data = active_sampling(
            labeled_data=cache.replay(), unlabeled_data=tr_data,
            number=(al_ratio / epoch_num) if seq_name != 0 else len(drift_pool),
            strategy='random', model=model, device=device)

        for epoch in range(1, epoch_num + 1):
            if seq_name != 0:
                select_data.extend(active_sampling(
                    labeled_data=[*cache.replay(), *select_data],
                    unlabeled_data=drift_pool, number=al_ratio / epoch_num,
                    strategy=al_strategy, model=model,
                    device=device))

            memory_dataset = ClassificationDataset(cache.replay())
            drift_loader = DataLoader(
                ClassificationDataset(select_data),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )

            tr_ans = engine.train_one_epoch(
                model, _prev_model, optim, drift_loader, memory_dataset,
                device, epoch, epoch_num)
            exp_ans['retrain'][f'{seq_name}'].append(tr_ans)

        # rehearsal memory update
        drift_loader = DataLoader(
            ClassificationDataset(select_data),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        cache.insert(model, drift_loader, device)
        cache.reduce()

        # after evaluation
        eval_ans = engine.evaluate(model, val_loader, device)
        exp_ans['test'][f'{seq_name}-after'].append(eval_ans)

    return exp_ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--core50_root', type=str, required=True)
    parser.add_argument('--core50_cumul', type=int, required=False, default=8)
    parser.add_argument('--core50_category', type=int, required=False, default=50)

    parser.add_argument('--model', type=str, required=False, default='resnet18')
    parser.add_argument('--num_classes', type=int, required=False, default=50)
    parser.add_argument('--pretrained_weights', type=str, required=False, default=None)

    parser.add_argument('--al_strategy', type=str, required=False, default='coreset')
    parser.add_argument('--al_ratio', type=float, required=False, default=0.1)
    parser.add_argument('--cache_max', type=int, required=False, default=500)

    parser.add_argument('--epoch_num', type=int, required=False, default=10)
    parser.add_argument('--lr', type=float, required=False, default=1e-3)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-5)

    parser.add_argument('--result_dir', type=str, required=False, default='./')
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--num_workers', type=int, required=False, default=0)
    parser.add_argument('--seed', type=int, required=False, default=42069)
    parser.add_argument('--device', type=str, required=False, default='cuda')

    args = vars(parser.parse_args())
    exp_ans = run_exp(
        args['core50_root'], args['core50_cumul'], args['core50_category'],
        args['model'], args['num_classes'], args['pretrained_weights'],
        args['al_strategy'], args['al_ratio'], args['cache_max'],
        args['epoch_num'], args['lr'], args['weight_decay'],
        args['batch_size'], args['num_workers'], args['seed'], args['device'])

    ans = {'setting': args, **exp_ans}
    print(ans)

    save_path = os.path.join(
        args['result_dir'], f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    with open(f'{save_path}.json', "w") as f:
        json.dump(ans, f, indent=2)
