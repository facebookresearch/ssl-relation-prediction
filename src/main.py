# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import sys
import ast
import argparse
from engines import KBCEngine


datasets = ['FB15K-237', 'WN18RR', 'aristo-v4',
            'UMLS', 'KINSHIP', 'NATIONS',
            'ogbl-biokg', 'ogbl-wikikg2']

parser = argparse.ArgumentParser(
    description="Relation Prediction as an Auxiliary Training Objective"
)

parser.add_argument(
    '--alias', default='',
    help='Alias for the experiments'
)
parser.add_argument(
    '--experiment_id', default='',
    help='Experiment ID which current run belongs to'
)
parser.add_argument(
    '--run_tags', default=[], type=ast.literal_eval,
    help='Tags for current run'
)
parser.add_argument(
    '--run_notes', default='', 
)
parser.add_argument(
    '--seed', default=0, type=str,
    help='For significance test'
)
parser.add_argument(
    '--device', default='cuda', type=str,
    help='Cuda or CPU'
)
parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)
parser.add_argument('--reciprocal', type=ast.literal_eval, default=True)
models = ['CP', 'ComplEx', 'TransE', 'RESCAL', 'TuckER']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--rank_r', default=100, type=int,
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale of the embeddings"
)
parser.add_argument(
    '--dropout', default=0, type=float,
    help="dropout"
)
parser.add_argument(
    '--lmbda', default=0, type=float,
    help="Regularization Strength"
)

worlds = ['LCWA', 'sLCWA+bpr', 'sLCWA+set']
parser.add_argument('--world', default='LCWA', choices=worlds, help="Training Approach + Loss"
)
parser.add_argument('--num_neg', type=int, default=0)
parser.add_argument('--score_rel', type=ast.literal_eval, default=False)
parser.add_argument('--score_lhs', type=ast.literal_eval, default=False)
parser.add_argument('--score_rhs', type=ast.literal_eval, default=True)
parser.add_argument('--w_rel', type=float, default=1)
parser.add_argument('--w_lhs', type=float, default=1)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=int,
    help="Number of epochs before doing validation."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

parser.add_argument(
    '--cache_eval', default=None, # './tmp/eval/{dataset}/{alias}/'
    help='whether or not to cache per evaluation result'
)
parser.add_argument(
    '--model_cache_path', default=None, # './tmp/model/{dataset}/{alias}/'
)


if __name__ == "__main__":
    args = parser.parse_args()
    engine_opt = vars(args)
    engine = KBCEngine(engine_opt)
    engine.episode()
    print('Episode Done', flush=True)
    sys.exit()