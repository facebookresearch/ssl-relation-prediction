# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import pickle
from typing import Tuple
from pathlib import Path

import torch
import numpy as np
from models import KBCModel

DATA_PATH = Path.cwd() / 'data'


class Sampler(object):
    """Sampler over the data. A sampler is dynamic pool while a dataset is a static array"""

    def __init__(self, data, n_ent, permute=True):
        """data: numpy array"""
        if permute:
            self.data = data[torch.randperm(data.shape[0]), :]
        else:
            self.data = data
        self.permute = permute
        self.size = len(data)
        self.n_ent = n_ent
        self._idx = 0
        self._epoch_idx = 0
        print('Creating a sampler of size {}'.format(self.size))

    def batchify(self, batch_size, device, num_neg=None):
        if self.is_empty():
            self.data = self.data[torch.randperm(self.data.shape[0]), :]
            self._idx = 0
            self._epoch_idx += 1
        if num_neg is None:
            batch = self.data[self._idx: self._idx + batch_size].to(device)
            self._idx = self._idx + batch_size
            return batch
        else:
            batch_size = int(batch_size / (2 * num_neg))
            pos_batch = self.data[self._idx: self._idx + batch_size]
            pos_batch = pos_batch.repeat(num_neg, 1).to(device)
            neg_batch = pos_batch.clone() 
            n = pos_batch.shape[0] # batch_size * num_neg
            neg_entity = torch.randint(high=self.n_ent - 1, low=0, size=(n,), device=device)
            neg_batch[:, 2] = neg_entity 
            label = torch.ones(n, 1).to(device)
            self._idx = self._idx + batch_size
            return pos_batch, neg_batch, label

    def is_empty(self):
        return (self._idx >= self.size)

    def is_epoch(self, epoch_idx):
        return (self._epoch_idx == epoch_idx)


class Dataset(object):
    def __init__(self, opt, data_path=None):
        self.opt = opt
        self.name = opt['dataset']
        self.device = opt['device']
        self.reciprocal = opt['reciprocal']
        if data_path is None:
            self.root = DATA_PATH / self.name
        else:
            self.root = Path(data_path)

        self.data = {}
        self.splits = ['train', 'valid', 'test']
        for f in self.splits:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        if self.reciprocal:
            self.n_predicates *= 2

        with open(str(self.root / f'to_skip.pickle'), 'rb') as inp_f:
            self.to_skip = pickle.load(inp_f) 

        print('{} Dataset Stat: {}'.format(self.name, self.get_shape()))

        n_train = len(self.get_examples('train')) 
        n_valid = len(self.get_examples('valid'))
        n_test = len(self.get_examples('test'))
        print('Train/Valid/Test {}/{}/{}'.format(n_train, n_valid, n_test))
        tot = 1.0 * (n_train + n_valid + n_test)
        print('Train/Valid/Test {:.3f}/{:.3f}/{:.3f}'.format(n_train / tot,
                                                             n_valid / tot,
                                                             n_test / tot))
        self.examples_train = torch.from_numpy(self.get_split(split='train'))
        self.examples_valid = torch.from_numpy(self.get_split(split='valid'))

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities

    def get_examples(self, split): 
        """ original split without reciprocal
        """
        return self.data[split].astype('int64')

    def get_split(self, split='train', reciprocal=True, unified_vocab=False):
        """ processed split with reciprocal & unified vocabuary

        Args:
            reciprocal: bool, whether to include reciprocal triples
            unified_vocab: bool, whether to use a unified vocabulary for entity + predicate
                           instead of separate vocabularies for entity and predicate
        """
        if self.reciprocal:
            copy = np.copy(self.data[split])
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            data = np.vstack((self.data[split], copy))
        else:
            data = self.data[split]
        return data.astype('int64')

    def get_sampler(self, split):
        examples = {'train': self.examples_train,
                    'valid': self.examples_valid}[split]
        sampler = Sampler(examples, 
                          self.n_entities)
        return sampler

    def eval(self,
             model: KBCModel, split: str,
             n_queries: int = -1, 
             n_epochs: int = -1,
             missing_eval: str = 'both', at: Tuple[int] = (1, 3, 10)):
        query_type = missing_eval 
        print('Evaluate the split {}'.format(split))
        test = self.get_examples(split)
        examples = torch.from_numpy(test).to(self.device)

        query_types = ['rhs', 'lhs'] if query_type == 'both' else [query_type]

        res, mean_reciprocal_rank, hits_at = {}, {}, {}
        for m in query_types:
            q = examples.clone()
            if n_queries > 0:  # used to sample a subset of train, 
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            candidates = m
            if m == 'lhs': 
                if self.reciprocal:
                    tmp = torch.clone(q[:, 0])
                    q[:, 0] = q[:, 2]
                    q[:, 2] = tmp
                    q[:, 1] += self.n_predicates // 2  # index for inverse predicate
                    candidates = 'rhs' # after reversing, the candidates to score are at rhs
            ranks, predicted = model.get_ranking(q, self.to_skip[m], batch_size=2000, candidates=candidates)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))
            res[m] = {'query': examples,  # triples to compute rhs raking among all the entities
                      'rank': ranks,
                      'predicted': predicted}
        return mean_reciprocal_rank, hits_at, res


    def get_scope(self, split): #TODO: this can be read from to_skip & remove irrelevant splits
        """get scope for each s-p-{} and {}-p-o"""
        if split == 'train':
            X = self.get_examples('train')
        elif split == 'train+valid':
            X = np.concatenate((self.get_examples('train'), self.get_examples('valid')),
                               axis=0)
        sp_to_o = dict()
        po_to_s = dict()

        for s, p, o in X:
            if (s, p) not in sp_to_o:
                sp_to_o[(s, p)] = []
            if (p, o) not in po_to_s:
                po_to_s[(p, o)] = []

            sp_to_o[(s, p)] += [o]
            po_to_s[(p, o)] += [s]
        return sp_to_o, po_to_s
