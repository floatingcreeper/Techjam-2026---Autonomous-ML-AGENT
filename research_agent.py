"""Guarded public-validation experiment controller for KuaiRand-Pure.

The controller executes named, reproducible experiments only.  It never asks a
model runner to load or score the test split.  Each attempt appends a JSONL
record containing its hypothesis, configuration, metrics, elapsed time, code
diff summary, and recovery details.
"""
import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass

from bpr_fm import run_bpr_fm
from blend_experiment import run_blend
from data import load_public
from sequence_ranker import run_sequence_ranker


EPSILON = 0.002
CONVERGENCE_N = 3


@dataclass(frozen=True)
class Experiment:
    name: str
    hypothesis: str
    config: dict


REGISTRY = (
    Experiment(
        name='bpr_fm',
        hypothesis='Pairwise same-user training should align FM with GAUC and nDCG@5.',
        config={'seed': 0, 'k': 16, 'lr': 0.001, 'epochs': 24, 'negative_ratio': 1},
    ),
    Experiment(
        name='fm_din_mtl',
        hypothesis='Causal video-history attention plus retained FM crosses improves ranking beyond BPR-FM.',
        config={'seed': 0, 'history_length': 20, 'embedding_dim': 8, 'learning_rate': 0.0003,
                'epochs': 5, 'batch_size': 4096, 'pair_weight': 0.5, 'auxiliary_weight': 0.04,
                'dropout': 0.25, 'weight_decay': 1e-5},
    ),
    Experiment(
        name='fm_din_mtl_seed1',
        hypothesis='The FM-DIN-MTL improvement should reproduce under a second random seed.',
        config={'seed': 1, 'history_length': 20, 'embedding_dim': 8, 'learning_rate': 0.0003,
                'epochs': 5, 'batch_size': 4096, 'pair_weight': 0.5, 'auxiliary_weight': 0.04,
                'dropout': 0.25, 'weight_decay': 1e-5},
    ),
    Experiment(
        name='fm_din_mtl_seed2',
        hypothesis='The FM-DIN-MTL improvement should reproduce under a third random seed.',
        config={'seed': 2, 'history_length': 20, 'embedding_dim': 8, 'learning_rate': 0.0003,
                'epochs': 5, 'batch_size': 4096, 'pair_weight': 0.5, 'auxiliary_weight': 0.04,
                'dropout': 0.25, 'weight_decay': 1e-5},
    ),
    Experiment(
        name='rank_blend',
        hypothesis='A within-user rank blend combines complementary BPR-FM and FM-DIN-MTL ordering errors.',
        config={'seed': 0},
    ),
)


def _diff_summary():
    try:
        diff = subprocess.check_output(
            ['git', 'diff', '--stat'], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ['git', 'status', '--short'], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return '\n'.join(part for part in (diff, 'Untracked/modified files:\n' + status if status else '') if part)
    except (OSError, subprocess.CalledProcessError):
        return ''


def run_experiment(experiment, data_dir):
    config = dict(experiment.config)
    if experiment.name == 'bpr_fm':
        splits = load_public(data_dir)
        return run_bpr_fm(
            splits, k=config.pop('k'), lr=config.pop('lr'), epochs=config.pop('epochs'),
            negative_ratio=config.pop('negative_ratio'), verbose=False, **config,
        )['valid']
    if experiment.name.startswith('fm_din_mtl'):
        return run_sequence_ranker(data_dir=data_dir, verbose=False, **config)[0]
    if experiment.name == 'rank_blend':
        return run_blend(data_dir=data_dir, **config)['blend']
    raise ValueError(f'unknown experiment {experiment.name!r}')


def run_agent(data_dir, iterations, log_path):
    history, best_primary, stagnant = [], float('-inf'), 0
    for iteration, experiment in enumerate(REGISTRY[:iterations]):
        started = time.time()
        record = {
            'iteration': iteration,
            'experiment': asdict(experiment),
            'code_diff': _diff_summary(),
        }
        try:
            metrics = run_experiment(experiment, data_dir)
            metrics = {name: float(value) if hasattr(value, 'item') else value
                       for name, value in metrics.items()}
            record.update({'status': 'ok', 'metrics': metrics})
            if metrics['primary'] > best_primary + EPSILON:
                best_primary = metrics['primary']
                stagnant = 0
            else:
                stagnant += 1
        except Exception as error:  # logged recovery event; following experiments still run
            record.update({'status': 'failed', 'error': repr(error)})
            stagnant += 1
        record['elapsed_sec'] = round(time.time() - started, 2)
        record['consecutive_non_improvements'] = stagnant
        history.append(record)
        with open(log_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False))
        # Complete the fixed initial portfolio before treating three small gains
        # as convergence; otherwise the registered blend would never be tried.
        if stagnant >= CONVERGENCE_N and iteration + 1 >= min(iterations, len(REGISTRY)):
            print(f'Converged after {CONVERGENCE_N} attempts without a > {EPSILON:.3f} gain.')
            break
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    parser.add_argument('--iterations', type=int, default=len(REGISTRY))
    parser.add_argument('--log', default='research_runs.jsonl')
    arguments = parser.parse_args()
    run_agent(arguments.data_dir, arguments.iterations, arguments.log)
