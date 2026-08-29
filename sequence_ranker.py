"""Causal DIN-style multi-task ranker for KuaiRand-Pure.

This module deliberately uses only train and public-validation data.  A training
example sees events earlier than its timestamp; a validation example sees the
completed training history only.  Current-impression feedback is used as a
training target, never as an input feature.
"""
import argparse
import csv
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bpr_fm import IntraUserPairSampler
from evaluate import evaluate


TRAIN_RANGE = (20220408, 20220421)
VALID_RANGE = (20220422, 20220428)
AUXILIARY_COLUMNS = ('is_click', 'is_like', 'is_follow', 'is_comment', 'is_profile_enter')


@dataclass
class SplitData:
    features: np.ndarray
    labels: np.ndarray
    auxiliary: np.ndarray
    users: np.ndarray
    timestamps: np.ndarray
    history_videos: np.ndarray
    history_feedback: np.ndarray


@dataclass
class SequenceData:
    train: SplitData
    valid: SplitData
    field_dims: tuple


def _split_for_date(date):
    if TRAIN_RANGE[0] <= date <= TRAIN_RANGE[1]:
        return 'train'
    if VALID_RANGE[0] <= date <= VALID_RANGE[1]:
        return 'valid'
    return None


def _iter_public_rows(data_dir):
    """Yield raw train/validation rows; test rows are skipped before labels are read."""
    author_by_video = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv'), newline='') as handle:
        for row in csv.DictReader(handle):
            author_by_video[row['video_id']] = row['author_id']

    for filename in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, filename), newline='') as handle:
            for row in csv.DictReader(handle):
                split = _split_for_date(int(row['date']))
                if split is None:
                    continue
                yield split, row, author_by_video.get(row['video_id'], 'UNK')


def _new_vocab():
    return {'<UNK>': 0}


def _add(vocab, value):
    if value not in vocab:
        vocab[value] = len(vocab)


def _feedback_code(row):
    code = 0
    for bit, column in enumerate(('long_view',) + AUXILIARY_COLUMNS):
        if row[column] != '0':
            code |= 1 << bit
    return code


def _write_history(target_videos, target_feedback, row_index, state, history_length):
    history = state
    if not history:
        return
    start = history_length - len(history)
    for offset, (video, feedback) in enumerate(history):
        target_videos[row_index, start + offset] = video
        target_feedback[row_index, start + offset] = feedback


def load_sequence_data(data_dir, history_length=20, duration_buckets=10):
    """Encode public data and attach causal user-history sequences."""
    vocabs = {name: _new_vocab() for name in ('user', 'video', 'author', 'tab', 'hour')}
    durations, counts = [], {'train': 0, 'valid': 0}
    for split, row, author in _iter_public_rows(data_dir):
        counts[split] += 1
        if split == 'train':
            _add(vocabs['user'], row['user_id'])
            _add(vocabs['video'], row['video_id'])
            _add(vocabs['author'], author)
            _add(vocabs['tab'], row['tab'])
            _add(vocabs['hour'], str(int(row['hourmin']) // 100))
            durations.append(float(row['duration_ms']))

    duration_edges = np.quantile(
        np.asarray(durations, dtype=np.float32),
        np.linspace(0, 1, duration_buckets + 1)[1:-1],
    )

    raw = {}
    for split in ('train', 'valid'):
        raw[split] = {
            'features': np.empty((counts[split], 6), dtype=np.int32),
            'labels': np.empty(counts[split], dtype=np.float32),
            'auxiliary': np.empty((counts[split], len(AUXILIARY_COLUMNS)), dtype=np.float32),
            'users': np.empty(counts[split], dtype=np.int32),
            'timestamps': np.empty(counts[split], dtype=np.int64),
            'feedback': np.empty(counts[split], dtype=np.uint8),
        }
    positions = {'train': 0, 'valid': 0}
    for split, row, author in _iter_public_rows(data_dir):
        position = positions[split]
        positions[split] += 1
        item = raw[split]
        user = vocabs['user'].get(row['user_id'], 0)
        video = vocabs['video'].get(row['video_id'], 0)
        item['features'][position] = (
            user,
            video,
            vocabs['author'].get(author, 0),
            vocabs['tab'].get(row['tab'], 0),
            int(np.searchsorted(duration_edges, float(row['duration_ms']))) + 1,
            vocabs['hour'].get(str(int(row['hourmin']) // 100), 0),
        )
        item['labels'][position] = 1.0 if row['long_view'] != '0' else 0.0
        item['auxiliary'][position] = [1.0 if row[column] != '0' else 0.0 for column in AUXILIARY_COLUMNS]
        item['users'][position] = user
        item['timestamps'][position] = int(row['time_ms'])
        item['feedback'][position] = _feedback_code(row)

    train_histories = np.zeros((counts['train'], history_length), dtype=np.int32)
    train_feedback = np.zeros((counts['train'], history_length), dtype=np.uint8)
    states = defaultdict(lambda: deque(maxlen=history_length))
    train_order = np.argsort(raw['train']['timestamps'], kind='stable')
    for index in train_order:
        user = int(raw['train']['users'][index])
        state = states[user]
        _write_history(train_histories, train_feedback, index, state, history_length)
        state.append((int(raw['train']['features'][index, 1]), int(raw['train']['feedback'][index])))

    valid_histories = np.zeros((counts['valid'], history_length), dtype=np.int32)
    valid_feedback = np.zeros((counts['valid'], history_length), dtype=np.uint8)
    for index, user in enumerate(raw['valid']['users']):
        _write_history(valid_histories, valid_feedback, index, states[int(user)], history_length)

    def split_data(name, histories, feedback_histories):
        item = raw[name]
        return SplitData(
            features=item['features'], labels=item['labels'], auxiliary=item['auxiliary'],
            users=item['users'], timestamps=item['timestamps'],
            history_videos=histories, history_feedback=feedback_histories,
        )

    return SequenceData(
        train=split_data('train', train_histories, train_feedback),
        valid=split_data('valid', valid_histories, valid_feedback),
        field_dims=(len(vocabs['user']), len(vocabs['video']), len(vocabs['author']),
                    len(vocabs['tab']), duration_buckets + 1, len(vocabs['hour'])),
    )


class DINMultiTaskRanker(nn.Module):
    """Candidate-aware attention over a user's prior video interactions."""

    def __init__(self, field_dims, embedding_dim=16, dropout=0.1):
        super().__init__()
        user_dim, video_dim, author_dim, tab_dim, duration_dim, hour_dim = field_dims
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(user_dim, embedding_dim, padding_idx=0)
        self.video_embedding = nn.Embedding(video_dim, embedding_dim, padding_idx=0)
        self.author_embedding = nn.Embedding(author_dim, embedding_dim, padding_idx=0)
        self.tab_embedding = nn.Embedding(tab_dim, embedding_dim, padding_idx=0)
        self.duration_embedding = nn.Embedding(duration_dim, embedding_dim, padding_idx=0)
        self.hour_embedding = nn.Embedding(hour_dim, embedding_dim, padding_idx=0)
        self.feedback_embedding = nn.Embedding(1 << (len(AUXILIARY_COLUMNS) + 1), embedding_dim, padding_idx=0)
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(size, 1, padding_idx=0)
            for size in (user_dim, video_dim, author_dim, tab_dim, duration_dim, hour_dim)
        ])

        input_dim = embedding_dim * 9
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 96), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(96, 48), nn.ReLU(),
        )
        self.primary_head = nn.Linear(48, 1)
        self.auxiliary_head = nn.Linear(48, len(AUXILIARY_COLUMNS))
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features, history_videos, history_feedback):
        user = self.user_embedding(features[:, 0])
        video = self.video_embedding(features[:, 1])
        author = self.author_embedding(features[:, 2])
        tab = self.tab_embedding(features[:, 3])
        duration = self.duration_embedding(features[:, 4])
        hour = self.hour_embedding(features[:, 5])

        history = self.video_embedding(history_videos) + self.feedback_embedding(history_feedback)
        attention_logits = (history * video.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.embedding_dim)
        history_mask = history_videos.ne(0)
        attention_logits = attention_logits.masked_fill(~history_mask, -1e9)
        attention = torch.softmax(attention_logits, dim=1)
        attention = attention * history_mask
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = (attention.unsqueeze(-1) * history).sum(dim=1)

        field_embeddings = (user, video, author, tab, duration, hour, interest)
        stacked = torch.stack(field_embeddings, dim=1)
        field_sum = stacked.sum(dim=1)
        fm_score = 0.5 * ((field_sum ** 2).sum(dim=1) - (stacked ** 2).sum(dim=(1, 2)))
        linear_score = sum(
            embedding(features[:, field]).squeeze(1)
            for field, embedding in enumerate(self.linear_embeddings)
        )
        representation = torch.cat((*field_embeddings, user * video, interest * video), dim=1)
        hidden = self.trunk(representation)
        primary = self.primary_head(hidden).squeeze(1) + fm_score + linear_score
        return primary, self.auxiliary_head(hidden)


def _torch_split(split):
    return {
        'features': torch.from_numpy(split.features),
        'labels': torch.from_numpy(split.labels),
        'auxiliary': torch.from_numpy(split.auxiliary),
        'history_videos': torch.from_numpy(split.history_videos),
        'history_feedback': torch.from_numpy(split.history_feedback.astype(np.int64, copy=False)),
    }


def _forward(model, tensors, indices):
    return model(
        tensors['features'].index_select(0, indices),
        tensors['history_videos'].index_select(0, indices),
        tensors['history_feedback'].index_select(0, indices),
    )


def predict(model, tensors, batch_size=8192):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(tensors['labels']), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(tensors['labels'])))
            scores.append(_forward(model, tensors, indices)[0].cpu().numpy())
    return np.concatenate(scores)


def run_sequence_ranker(data_dir='./KuaiRand-Pure/data', history_length=20, embedding_dim=16,
                        learning_rate=0.001, epochs=8, batch_size=2048, pair_weight=0.35,
                        auxiliary_weight=0.08, dropout=0.1, weight_decay=1e-6,
                        patience=3, seed=0, verbose=True, return_predictions=False):
    """Train and evaluate strictly on public validation data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    data = load_sequence_data(data_dir, history_length=history_length)
    train_tensors, valid_tensors = _torch_split(data.train), _torch_split(data.valid)
    model = DINMultiTaskRanker(data.field_dims, embedding_dim=embedding_dim, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sampler = IntraUserPairSampler(data.train.labels, data.train.users, seed=seed)

    best_primary, best_state, stale = -1.0, None, 0
    row_count = len(data.train.labels)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        row_order = torch.from_numpy(np.random.permutation(row_count).astype(np.int64))
        pos_indices, neg_indices = sampler.sample_pairs(negative_ratio=1)
        pos_indices = torch.from_numpy(pos_indices.astype(np.int64, copy=False))
        neg_indices = torch.from_numpy(neg_indices.astype(np.int64, copy=False))
        losses = []
        for start in range(0, row_count, batch_size):
            point_indices = row_order[start:start + batch_size]
            pair_start = start % len(pos_indices)
            pair_stop = min(pair_start + len(point_indices), len(pos_indices))
            pair_pos = pos_indices[pair_start:pair_stop]
            pair_neg = neg_indices[pair_start:pair_stop]
            if len(pair_pos) < len(point_indices):
                remaining = len(point_indices) - len(pair_pos)
                pair_pos = torch.cat((pair_pos, pos_indices[:remaining]))
                pair_neg = torch.cat((pair_neg, neg_indices[:remaining]))

            primary_logits, auxiliary_logits = _forward(model, train_tensors, point_indices)
            point_loss = F.binary_cross_entropy_with_logits(
                primary_logits, train_tensors['labels'].index_select(0, point_indices)
            )
            auxiliary_loss = F.binary_cross_entropy_with_logits(
                auxiliary_logits, train_tensors['auxiliary'].index_select(0, point_indices)
            )
            pos_scores = _forward(model, train_tensors, pair_pos)[0]
            neg_scores = _forward(model, train_tensors, pair_neg)[0]
            pair_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss = point_loss + auxiliary_weight * auxiliary_loss + pair_weight * pair_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        scores = predict(model, valid_tensors)
        metrics = evaluate(data.valid.users, data.valid.labels, scores)
        if verbose:
            print(f"  epoch {epoch:2d} | loss {np.mean(losses):.4f} | valid GAUC {metrics['GAUC']:.4f} "
                  f"nDCG@5 {metrics['nDCG@5']:.4f} primary {metrics['primary']:.4f} "
                  f"| {time.time() - started:.1f}s")
        if metrics['primary'] > best_primary + 1e-5:
            best_primary = metrics['primary']
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f'  early stop at epoch {epoch}')
                break

    model.load_state_dict(best_state)
    final_scores = predict(model, valid_tensors)
    final_metrics = evaluate(data.valid.users, data.valid.labels, final_scores)
    if return_predictions:
        return final_metrics, model, data, final_scores
    return final_metrics, model, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    parser.add_argument('--history_length', type=int, default=20)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--pair_weight', type=float, default=0.35)
    parser.add_argument('--auxiliary_weight', type=float, default=0.08)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print('Loading causal train/public-validation histories ...')
    metrics, _, _ = run_sequence_ranker(
        data_dir=args.data_dir, history_length=args.history_length,
        embedding_dim=args.embedding_dim, learning_rate=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, pair_weight=args.pair_weight,
        auxiliary_weight=args.auxiliary_weight, dropout=args.dropout,
        weight_decay=args.weight_decay, seed=args.seed,
    )
    print(f"\n=== DIN-MTL (seed={args.seed}) ===")
    print(f"  valid  GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} "
          f"| primary {metrics['primary']:.4f}")
