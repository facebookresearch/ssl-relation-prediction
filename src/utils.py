# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from torch import optim
import random

import numpy as np
import networkx as nx

from typing import Tuple, List, Dict, Set, Optional

import logging
import subprocess

logger = logging.getLogger(__name__)


def set_seed(seed: int, is_deterministic=True):
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def setup_optimizer(model, type, learning_rate, decay1, decay2, momentum=0):
    return {
        'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=learning_rate),
        # 'Adam': lambda: optim.Adam(model.parameters(), lr=learning_rate, betas=(decay1, decay2)),
        'Adam': lambda: optim.SparseAdam(model.parameters(), lr=learning_rate, betas=(decay1, decay2)),
        'SGD': lambda: optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    }[type]()


def avg_both(mrrs, hits):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
                    b += 1  # batch_idx
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    # print(mrrs)
    return {'MRR': m, 'hits@[1,3,10]': h.tolist()}


def get_avg_param(model):
    s = 0.0
    cnt = 0.0
    for param in model.parameters():
        s += param.sum()
        cnt += np.prod(param.shape)
    # print('s {}, cnt {}'.format(s, cnt))
    return s / cnt


def get_grad_norm(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            if p.grad.is_sparse:
                grads.append(p.grad.data.to_dense().view(-1, 1))
            else:
                grads.append(p.grad.data.view(-1, 1))
    if len(grads) == 0:
        grads.append(torch.FloatTensor([0]))
    grad_norm = torch.norm(torch.cat(grads))
    if grad_norm.is_cuda:
        grad_norm = grad_norm.cpu()
    return grad_norm.item()


def get_optimizer_status(optimizer):
    if 'Adagrad' in str(optimizer.__class__):
        optim_status = {'step': [v['step'] for _, v in optimizer.state.items() if len(v) > 0],
                   'sum': [v['sum'].data.detach() for _, v in optimizer.state.items() if len(v) > 0]}
        return optim_status
    # TODO: other optimziers
    return None


def to_networkx(triples: List[Tuple[str, str, str]],
                entity_to_idx: Dict[str, int],
                predicate_to_idx: Dict[str, int],
                predicates: Optional[Set[str]] = None,
                is_multidigraph: bool = False) -> nx.DiGraph:

    _triples = triples if predicates is None else [(s, p, o) for s, p, o in triples if p in predicates]

    G = nx.MultiDiGraph() if is_multidigraph else nx.DiGraph()

    entities = sorted({s for (s, _, _) in triples} | {o for (_, _, o) in triples})
    G.add_nodes_from([entity_to_idx[e] for e in entities])

    if is_multidigraph:
        G.add_edges_from([(entity_to_idx[s], entity_to_idx[o], {'p': predicate_to_idx[p]}) for s, p, o in _triples])
    else:
        edge_lst = sorted({(entity_to_idx[s], entity_to_idx[o]) for s, p, o in _triples})
        G.add_edges_from(edge_lst)

    return G

# @profile
def get_graph_features(triples: List[Tuple[str, str, str]],
                       entity_to_idx: Dict[str, int],
                       predicate_to_idx: Dict[str, int],
                       predicates: Optional[Set[str]] = None) -> np.ndarray:
    G = to_networkx(triples, entity_to_idx, predicate_to_idx, predicates, is_multidigraph=False)
    uG = G.to_undirected()

    mG = to_networkx(triples, entity_to_idx, predicate_to_idx, predicates, is_multidigraph=True)
    # umG = mG.to_undirected()

    logger.debug('mG.degree() ..')
    f1 = mG.degree()

    logger.debug('mG.in_degree() ..')
    f2 = mG.in_degree()

    logger.debug('mG.out_degree() ..')
    f3 = mG.out_degree()

    logger.debug('nx.pagerank(G) ..')
    f4 = nx.pagerank(G)

    logger.debug('nx.degree_centrality(mG) ..')
    f5 = nx.degree_centrality(mG)

    logger.debug('nx.in_degree_centrality(mG) ..')
    f6 = nx.in_degree_centrality(mG)

    logger.debug('nx.out_degree_centrality(mG) ..')
    f7 = nx.out_degree_centrality(mG)

    feature_lst = [f1, f2, f3, f4, f5, f6, f7]

    nb_entities = int(max(v for _, v in entity_to_idx.items()) + 1)
    nb_features = len(feature_lst)
    res = np.zeros(shape=(nb_entities, nb_features), dtype=np.float32) # TODO: predicate features

    for i, f in enumerate(feature_lst):
        for k, v in (f.items() if isinstance(f, dict) else f):
            res[k, i] = v
    res[:, :4] = np.log(res[:, :4] + 1e-7) # log degree
    return res


if __name__ == '__main__':
    triples = [
        ('a', 'p', 'b'),
        ('a', 'p', 'c'),
        ('b', 'q', 'd')
    ]

    entity_to_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    predicate_to_idx = {'p': 0, 'q': 1}

    features = get_graph_features(triples, entity_to_idx, predicate_to_idx)

    print(features)
    print(features.shape)