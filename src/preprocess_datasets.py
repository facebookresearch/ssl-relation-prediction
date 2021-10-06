# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import errno
import pickle
from pathlib import Path
from collections import defaultdict
from datasets import Dataset
import numpy as np


DATA_PATH = Path.cwd() / 'data'
print('DATA_PATH: {}'.format(DATA_PATH))


def prepare_dataset(path, name):
    """KBC dataset preprocessing. 
    1) Maps each entity and relation to a unique id
    2) Create a corresponding folder of `cwd/data/dataset`, with mapped train/test/valid files.
    3) Create `to_skip_lhs` & `to_skip_rhs` for filtered metrics
    4) Save the mapping `rel_id` & `ent_id` for analysis.

    Args:
        path: a path of a folder containing 3 tab-separated files, `train`, `valid` and `test`.
        name: name of the dataset
    """
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    n_relations = len(relations)
    n_entities = len(entities)
    print("{} entities and {} relations".format(n_entities, n_relations))

    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids # TODO: improve using np.vectorize & dict.get
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files 
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for pos, skip in to_skip.items():
        for query, ans in skip.items():
            to_skip_final[pos][query] = sorted(list(ans))

    with open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb') as out:
        pickle.dump(to_skip_final, out)
    print('Done processing!')


if __name__ == "__main__":
    # datasets = ['FB15K-237', 'WN18RR', 'aristo-v4',
    #             'UMLS', 'KINSHIP', 'NATIONS']
    datasets = ['FB15K-237']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            p = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                             'src_data', d)
            prepare_dataset(p, d)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise

