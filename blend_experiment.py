"""Validation-only rank ensemble for the BPR-FM and FM-DIN-MTL models."""
import argparse
import collections

import numpy as np

from bpr_fm import run_bpr_fm
from data import load_public
from evaluate import evaluate
from sequence_ranker import run_sequence_ranker


def within_user_percentiles(users, scores):
    """Make model scales comparable while preserving each user's ranking."""
    result = np.zeros(len(scores), dtype=np.float32)
    groups = collections.defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    for indices in groups.values():
        order = sorted(indices, key=lambda index: scores[index])
        denominator = max(1, len(order) - 1)
        for rank, index in enumerate(order):
            result[index] = rank / denominator
    return result


def run_blend(data_dir='./KuaiRand-Pure/data', seed=0):
    splits = load_public(data_dir)
    bpr_metrics, bpr_scores = run_bpr_fm(
        splits, k=16, lr=0.001, epochs=24, negative_ratio=1, seed=seed,
        verbose=False, return_predictions=True,
    )
    din_metrics, _, sequence_data, din_scores = run_sequence_ranker(
        data_dir=data_dir, history_length=20, embedding_dim=8, learning_rate=0.0003,
        epochs=5, batch_size=4096, pair_weight=0.5, auxiliary_weight=0.04,
        dropout=0.25, weight_decay=1e-5, seed=seed, verbose=False,
        return_predictions=True,
    )
    if len(bpr_scores) != len(din_scores):
        raise ValueError('model predictions do not align')

    bpr_rank = within_user_percentiles(sequence_data.valid.users, bpr_scores)
    din_rank = within_user_percentiles(sequence_data.valid.users, din_scores)
    candidates = []
    for bpr_weight in np.linspace(0.0, 1.0, 11):
        metrics = evaluate(
            sequence_data.valid.users, sequence_data.valid.labels,
            bpr_weight * bpr_rank + (1.0 - bpr_weight) * din_rank,
        )
        candidates.append((metrics['primary'], float(bpr_weight), metrics))
    _, best_weight, best_metrics = max(candidates, key=lambda candidate: candidate[0])
    return {'bpr': bpr_metrics['valid'], 'fm_din_mtl': din_metrics,
            'blend': best_metrics, 'bpr_weight': best_weight}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    results = run_blend(args.data_dir, seed=args.seed)
    for name in ('bpr', 'fm_din_mtl', 'blend'):
        metrics = results[name]
        print(f"{name:10s} GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} "
              f"| primary {metrics['primary']:.4f}")
    print(f"blend BPR weight: {results['bpr_weight']:.1f}")
