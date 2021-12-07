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
import numpy as np

from ogb.linkproppred import LinkPropPredDataset

DATA_PATH = Path.cwd() / 'data'
print('DATA_PATH: {}'.format(DATA_PATH))


def prepare_dataset_ogb_wikikg2(name):
    """ogbl-wikikg2 is a OGB link property prediction dataset. 
    Note that the evaluation protocol is different from conventional KBC datasets.

    training input: (h,r,t)
    valid/test input: (h, r, t, h_neg, t_neg), including 500 negatives respectively for h and t.
    """
    dataset = LinkPropPredDataset(name)
    split_edge = dataset.get_edge_split()
    train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

    nrelation = int(max(train_triples['relation']))+1
    nentity = int(max(np.concatenate((train_triples['head'], 
                                      train_triples['tail']))))+1
    print(nentity, nrelation)

    train_array = np.concatenate((train_triples['head'].reshape(-1, 1),
                                  train_triples['relation'].reshape(-1, 1),
                                  train_triples['tail'].reshape(-1, 1),
                                  ), axis=1)

    valid_array = np.concatenate((valid_triples['head'].reshape(-1, 1),
                                  valid_triples['relation'].reshape(-1, 1),
                                  valid_triples['tail'].reshape(-1, 1),
                                  valid_triples['head_neg'],
                                  valid_triples['tail_neg'],
                                  ), axis=1)

    test_array = np.concatenate((test_triples['head'].reshape(-1, 1),
                                  test_triples['relation'].reshape(-1, 1),
                                  test_triples['tail'].reshape(-1, 1),
                                  test_triples['head_neg'],
                                  test_triples['tail_neg'],
                                  ), axis=1)
    print('Saving arrays ...')
    p = Path(DATA_PATH / name)
    p.mkdir(parents=True, exist_ok=True) 
    # using npy since it is too big for pickling
    with open(Path(DATA_PATH) / name / ('train' + '.npy'), 'wb') as out:
        np.save(out, train_array.astype('uint64')) 
    with open(Path(DATA_PATH) / name / ('valid' + '.npy'), 'wb') as out:
        np.save(out, valid_array.astype('uint64'))
    with open(Path(DATA_PATH) / name / ('test' + '.npy'), 'wb') as out:
        np.save(out, test_array.astype('uint64'))
    print('Saving meta_info ...')
    meta_info = {'n_provided_neg_head': valid_triples['head_neg'].shape[1],
                 'n_provided_neg_tail': valid_triples['tail_neg'].shape[1],
                 }
    out = open(Path(DATA_PATH) / name / ('meta_info' + '.pickle'), 'wb')
    pickle.dump(meta_info, out)
    out.close()
    print('Done processing!')


def prepare_dataset_ogb_biokg(name):
    """ogbl-biokg is a OGB link property prediction dataset
    Note that the input formats and evaluation protocols are different from conventional KBC datasets

    training input: (h,r,t, h_type, t_type), the last 2 indices represent the entity types.
    valid/test input: (h,r,t, h_neg, t_neg, h_type, t_type), including 500 negatives respectively for h and t
    """
    dataset = LinkPropPredDataset(name)
    split_edge = dataset.get_edge_split()
    train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]
    cur_idx, cur_type_idx, type_dict, entity_dict = 0, 0, {}, {}
    for key in dataset[0]['num_nodes_dict']:
        type_dict[key] = cur_type_idx
        cur_type_idx += 1
        entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
        cur_idx += dataset[0]['num_nodes_dict'][key]

    def index_triples_across_type(triples, entity_dict, type_dict):
        triples['head_type_idx'] = np.zeros_like(triples['head'])
        triples['tail_type_idx'] = np.zeros_like(triples['tail'])
        for i in range(len(triples['head'])):
            h_type = triples['head_type'][i]
            triples['head_type_idx'][i] = type_dict[h_type] 
            triples['head'][i] += entity_dict[h_type][0]
            if 'head_neg' in triples:
                triples['head_neg'][i] += entity_dict[h_type][0]
            t_type = triples['tail_type'][i]
            triples['tail_type_idx'][i] = type_dict[t_type]
            triples['tail'][i] += entity_dict[t_type][0]
            if 'tail_neg' in triples:
                triples['tail_neg'][i] += entity_dict[t_type][0]
        return triples
    
    print('Indexing triples across different entity types ...')
    train_triples = index_triples_across_type(train_triples, entity_dict, type_dict)
    valid_triples = index_triples_across_type(valid_triples, entity_dict, type_dict)
    test_triples = index_triples_across_type(test_triples, entity_dict, type_dict)
    nrelation = int(max(train_triples['relation']))+1
    nentity = sum(dataset[0]['num_nodes_dict'].values())
    assert train_triples['head'].max() <= nentity

    train_array = np.concatenate((train_triples['head'].reshape(-1, 1),
                                  train_triples['relation'].reshape(-1, 1),
                                  train_triples['tail'].reshape(-1, 1),
                                  train_triples['head_type_idx'].reshape(-1, 1),
                                  train_triples['tail_type_idx'].reshape(-1, 1),
                                  ), axis=1)

    valid_array = np.concatenate((valid_triples['head'].reshape(-1, 1),
                                  valid_triples['relation'].reshape(-1, 1),
                                  valid_triples['tail'].reshape(-1, 1),
                                  valid_triples['head_neg'],
                                  valid_triples['tail_neg'],
                                  valid_triples['head_type_idx'].reshape(-1, 1),
                                  valid_triples['tail_type_idx'].reshape(-1, 1),
                                  ), axis=1)

    test_array = np.concatenate((test_triples['head'].reshape(-1, 1),
                                  test_triples['relation'].reshape(-1, 1),
                                  test_triples['tail'].reshape(-1, 1),
                                  test_triples['head_neg'],
                                  test_triples['tail_neg'],
                                  test_triples['head_type_idx'].reshape(-1, 1),
                                  test_triples['tail_type_idx'].reshape(-1, 1),
                                  ), axis=1)
    print('Saving arrays ...')
    p = Path(DATA_PATH / name)
    p.mkdir(parents=True, exist_ok=True)
    with open(Path(DATA_PATH) / name / ('train' + '.pickle'), 'wb') as out:
        pickle.dump(train_array.astype('uint64'), out)

    with open(Path(DATA_PATH) / name / ('valid' + '.pickle'), 'wb') as out:
        pickle.dump(valid_array.astype('uint64'), out)

    with open(Path(DATA_PATH) / name / ('test' + '.pickle'), 'wb') as out:
        pickle.dump(test_array.astype('uint64'), out)

    print('Saving meta_info ...')
    meta_info = {'n_provided_neg_head': valid_triples['head_neg'].shape[1],
                 'n_provided_neg_tail': valid_triples['tail_neg'].shape[1],
                 'type_dict': type_dict, # type name: idx
                 'entity_dict': entity_dict, #  type name: first entity idx, last entity idx
                 }
    out = open(Path(DATA_PATH) / name / ('meta_info' + '.pickle'), 'wb')
    pickle.dump(meta_info, out)
    out.close()
    print('Done processing!')


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

    # map train/test/valid with the ids
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
    # datasets = ['FB15K-237']
    datasets = ['ogbl-wikikg2']
    datasets = ['ogbl-biokg']
    datasets = ['custom_graph']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            if d in ['ogbl-biokg']:
                prepare_dataset_ogb_biokg(d)
            elif d in ['ogbl-wikikg2']:
                prepare_dataset_ogb_wikikg2(d)
            else:
                p = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                'src_data', d)
                prepare_dataset(p, d)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise

