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
from ogb.linkproppred import Evaluator
from models import KBCModel

DATA_PATH = Path.cwd() / 'data'


def subsample(triples, n):
    """Subsample n entries from triples"""
    perm = torch.randperm(len(triples))[:n]
    q = triples[perm]
    return q


def invert(triples: np.array, n_rel: int, stack: bool=True, include_type=True):
    """Given triples, return the version containing reciprocal triples, used in training
    
    Args: 
        triples: h, r, t, h_neg, t_neg, h_type, t_type
        n_rel: the number of original relations
    """
    copy = np.copy(triples)
    tmp = np.copy(copy[:, 0])
    copy[:, 0] = copy[:, 2]
    copy[:, 2] = tmp
    copy[:, 1] += n_rel
    if include_type: # h,r,t,...h_type,t_type
        tmp = np.copy(copy[:, -1]) 
        copy[:, -1] = copy[:, -2]
        copy[:, -2] = tmp
    if stack:
        return np.vstack((triples, copy))
    else:
        return copy


def invert_torch(triples: torch.Tensor, n_rel: int, include_type=True):
    """Given triples, return the version containing reciprocal triples, used in valid/test
    
    Args: 
        triples: h, r, t, h_neg, t_neg, h_type, t_type
        n_rel: the number of original relations
    """
    tmp = torch.clone(triples[:, 0])
    triples[:, 0] = triples[:, 2]
    triples[:, 2] = tmp
    triples[:, 1] += n_rel
    del tmp
    if include_type:
        tmp = torch.clone(triples[:, -1]) 
        triples[:, -1] = triples[:, -2]
        triples[:, -2] = tmp  
        num_neg = (triples.shape[1] - 5) // 2
    else:
        num_neg = (triples.shape[1] - 3) // 2
    print('Num neg per head/tail {}'.format(num_neg))
    if num_neg > 0:
        tmp = torch.clone(triples[:, 3:3+num_neg])
        assert tmp.shape[1] == num_neg
        triples[:, 3:3+num_neg] = triples[:, 3+num_neg:3+2*num_neg]
        triples[:, 3+num_neg:3+2*num_neg] = tmp
        del tmp
    return triples


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
            p = str(self.root / (f + '.pickle'))
            if os.path.isfile(p):
                with open(p, 'rb') as in_file:
                    self.data[f] = pickle.load(in_file)
            else:
                p = str(self.root / (f + '.npy'))
                with open(p, 'rb') as in_file:
                    self.data[f] = np.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.include_type = self.name in ['ogbl-biokg'] # self.data['train'].shape[1] == 5
        self.bsz_vt = 16 if self.name in ['ogbl-wikikg2'] else 1000
        if self.reciprocal:
            self.n_predicates *= 2

        if os.path.isfile(str(self.root / 'to_skip.pickle')):
            print('Loading to_skip file ...')    
            with open(str(self.root / f'to_skip.pickle'), 'rb') as inp_f:
                self.to_skip = pickle.load(inp_f) # {'lhs': {(11, 3): [1, 3, 0, 4, 5, 19]}}

        if os.path.isfile(str(self.root / 'meta_info.pickle')):
            print('Loading meta_info file ...')
            with open(str(self.root / f'meta_info.pickle'), 'rb') as inp_f:
                self.meta_info = pickle.load(inp_f)  

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
        """ raw data without any processing
        """
        return self.data[split].astype('int64')

    def get_split(self, split='train', reciprocal=True):
        """ processed split with reciprocal & unified vocabuary

        Args:
            reciprocal: bool, whether to include reciprocal triples
        """
        data = self.data[split]
        if self.reciprocal:
            assert split != 'test'
            data = invert(data, self.n_predicates // 2, stack=True, 
                          include_type=self.include_type)
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
             query_type: str = 'both', at: Tuple[int] = (1, 3, 10)):
        print('Evaluate the split {}'.format(split))
        test = self.get_examples(split)
        examples = torch.from_numpy(test).to(self.device)
        query_types = ['rhs', 'lhs'] if query_type == 'both' else [query_type]
        res, mean_reciprocal_rank, hits_at = {}, {}, {}
        for m in query_types:
            print('Evaluating the {}'.format(m))
            q = examples.clone()
            if n_queries > 0:  # used to sample a subset of train, 
                q = subsample(examples, n_queries)
            candidate_pos = m
            if m == 'lhs': 
                if self.reciprocal:
                    q = invert_torch(q, self.n_predicates // 2, include_type=self.include_type)
                    candidate_pos = 'rhs' # after reversing, the candidates to score are at rhs
            if 'ogb' in self.name:
                evaluator = Evaluator(name=self.name)
                metrics = model.get_metric_ogb(q, 
                                               batch_size=self.bsz_vt, 
                                               query_type=candidate_pos, 
                                               evaluator=evaluator)
                mean_reciprocal_rank[m] = metrics['mrr_list']
                hits_at[m] = torch.FloatTensor([metrics['hits@{}_list'.format(k)] for k in at]) 
                res = None
            else:
                ranks, predicted = model.get_ranking(q, self.to_skip[m], 
                                                     batch_size=self.bsz_vt, 
                                                     candidates=candidate_pos)
                mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
                hits_at[m] = torch.FloatTensor((list(map(
                    lambda x: torch.mean((ranks <= x).float()).item(),
                    at
                ))))
                res[m] = {'query': examples,  # triples to compute rhs raking among all the entities
                          'rank': ranks,
                          'predicted': predicted}
            del q
        return mean_reciprocal_rank, hits_at, res


