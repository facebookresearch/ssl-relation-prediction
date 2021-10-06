# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from torch import nn

import os
from datasets import Dataset
from models import CP, ComplEx, TransE, RESCAL, TuckER
from regularizers import F2, N3
from utils import avg_both, setup_optimizer, get_git_revision_hash, set_seed

import wandb


def setup_ds(opt):
    dataset_opt = {k: v for k, v in opt.items() if k in ['dataset', 'device', 'cache_eval', 'reciprocal']}
    dataset = Dataset(dataset_opt)
    return dataset


def setup_model(opt):
    if opt['model'] == 'TransE':
        model = TransE(opt['size'], opt['rank'], opt['init'])
    elif opt['model'] == 'ComplEx':
        model = ComplEx(opt['size'], opt['rank'], opt['init'])
    elif opt['model'] == 'TuckER':
        model = TuckER(opt['size'], opt['rank'], opt['rank_r'], opt['init'], opt['dropout'])
    elif opt['model'] == 'RESCAL':
        model = RESCAL(opt['size'], opt['rank'], opt['init'])
    elif opt['model'] == 'CP':
        model = CP(opt['size'], opt['rank'], opt['init'])
    model.to(opt['device'])
    return model


def setup_loss(opt):
    if opt['world'] == 'sLCWA+bpr':
        loss = nn.BCEWithLogitsLoss(reduction='mean')
    elif opt['world'] == 'sLCWA+set':
        pass
    elif opt['world'] == 'LCWA':
        loss = nn.CrossEntropyLoss(reduction='mean')
    return loss


def setup_regularizer(opt):
    if opt['regularizer'] == 'F2':
        regularizer =  F2(opt['lmbda'])
    elif opt['regularizer'] == 'N3':
        regularizer = N3(opt['lmbda'])
    return regularizer


def _set_exp_alias(opt):
    suffix = '{}_{}_Rank{}_Reg{}_Lmbda{}'.format(opt['dataset'], opt['model'], opt['rank'], opt['regularizer'], opt['lmbda'])
    alias = opt['alias'] + suffix
    return alias


def _set_cache_path(path_template, dataset, alias):
    if path_template is not None:
        cache_path = path_template.format(dataset=dataset, alias=alias)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
    else:
        cache_path = None
    return cache_path


class KBCEngine(object):
    def __init__(self, opt):
        self.seed = opt['seed']
        set_seed(int(self.seed))
        self.alias = _set_exp_alias(opt)
        self.cache_eval = _set_cache_path(opt['cache_eval'], opt['dataset'], self.alias)
        self.model_cache_path = _set_cache_path(opt['model_cache_path'], opt['dataset'], self.alias)
        opt['cache_eval'] = self.cache_eval
        # dataset
        self.dataset = setup_ds(opt)
        opt['size'] = self.dataset.get_shape()
        # model
        self.model = setup_model(opt)
        self.optimizer = setup_optimizer(self.model, opt['optimizer'], opt['learning_rate'], opt['decay1'], opt['decay2'])
        self.loss = setup_loss(opt)
        opt['loss'] = self.loss
        self.batch_size = opt['batch_size']
        # regularizer
        self.regularizer = setup_regularizer(opt)
        self.device = opt['device']
        self.max_epochs = opt['max_epochs']
        self.world = opt['world']
        self.num_neg = opt['num_neg']
        self.score_rel = opt['score_rel']
        self.score_rhs = opt['score_rhs']
        self.score_lhs = opt['score_lhs']
        self.w_rel = opt['w_rel']
        self.w_lhs = opt['w_lhs']
        self.opt = opt
        self._epoch_id = 0

        wandb.init(project="ssl-relation-prediction", 
                    group=opt['experiment_id'], 
                    tags=opt['run_tags'],
                    notes=opt['run_notes'])
        wandb.config.update(opt)
        wandb.watch(self.model, log='all', log_freq=10000)
        wandb.run.summary['is_done'] = False
        print('Git commit ID: {}'.format(get_git_revision_hash()))
        
    def episode(self):
        best_valid_mrr, init_epoch_id, step_idx = 0, 0, 0
        exp_train_sampler = self.dataset.get_sampler('train')
        
        for e in range(init_epoch_id, self.max_epochs):
            wandb.run.summary['epoch_id'] = e
            self.model.train()
            while exp_train_sampler.is_epoch(e): # iterate through all batchs inside an epoch
                # 1 theta update
                if self.world == 'LCWA':
                    input_batch_train = exp_train_sampler.batchify(self.batch_size,
                                                                    self.device)
                    predictions, factors = self.model.forward(input_batch_train, score_rel=self.score_rel, score_rhs=self.score_rhs, score_lhs=self.score_lhs)
                    
                    if self.score_rel and self.score_rhs and self.score_lhs:
                        # print('----1----')
                        l_fit = self.loss(predictions[0], input_batch_train[:, 2]) \
                                + self.w_rel * self.loss(predictions[1], input_batch_train[:, 1]) \
                                + self.w_lhs * self.loss(predictions[2], input_batch_train[:, 0])
                    elif self.score_rel and self.score_rhs:
                        # print('----2----')
                        l_fit = self.loss(predictions[0], input_batch_train[:, 2]) + self.w_rel * self.loss(predictions[1], input_batch_train[:, 1])
                    elif self.score_lhs and self.score_rel:
                        # print('----3----')
                        pass
                    elif self.score_rhs and self.score_lhs: # standard
                        # print('----4----')
                        l_fit = self.loss(predictions[0], input_batch_train[:, 2]) + self.loss(predictions[1], input_batch_train[:, 0])
                    elif self.score_rhs: # only rhs
                        # print('----5----')
                        l_fit = self.loss(predictions, input_batch_train[:, 2])
                    elif self.score_rel:
                        # print('----6----')
                        l_fit = self.loss(predictions, input_batch_train[:, 1])
                    elif self.score_lhs:
                        # print('----7----')
                        pass
                    
                    l_reg, l_reg_raw, avg_lmbda = self.regularizer.penalty(input_batch_train, factors) # Note: this shouldn't be included into the computational graph of lambda update
                elif self.world == 'sLCWA+bpr':
                    pos_train, neg_train, label = exp_train_sampler.batchify(self.batch_size,
                                                                                self.device,
                                                                                num_neg=self.num_neg)
                    predictions, factors = self.model.forward_bpr(pos_train, neg_train)
                    l_fit = self.loss(predictions, label)
                    l_reg, l_reg_raw, avg_lmbda = self.regularizer.penalty(
                        torch.cat((pos_train, neg_train), dim=0),
                        factors)
                l = l_fit + l_reg
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                    
                if ((step_idx % 1000 == 0 and step_idx > 1000) or (step_idx <= 1000 and step_idx % 100 == 0)): # reduce logging frequency to accelerate 
                    wandb.log({'step_wise/train/l': l.item()}, step=step_idx)
                    wandb.log({'step_wise/train/l_fit': l_fit.item()}, step=step_idx)
                    wandb.log({'step_wise/train/l_reg': l_reg.item()}, step=step_idx)
                    wandb.log({'step_wise/train/l_reg_raw': l_reg_raw.item()}, step=step_idx)
                step_idx += 1
            if e % self.opt['valid'] == 0:
                self.model.eval()
                res_all, res_all_detailed = [], []
                for split in self.dataset.splits:
                    res_s = self.dataset.eval(model=self.model, 
                                              split=split, 
                                              n_queries=-1 if split != 'train' else 5000, # subsample 5000 triples for computing approximated training MRR
                                              n_epochs=e)
                    res_all.append(avg_both(res_s[0], res_s[1]))
                    res_all_detailed.append(res_s[2])
                    
                res = dict(zip(self.dataset.splits, res_all))
                res_detailed = dict(zip(self.dataset.splits, res_all_detailed))
                
                print("\t Epoch: ", e)
                for split in self.dataset.splits:
                    print("\t {}: {}".format(split.upper(), res[split]))
                    wandb.log({'step_wise/{}/mrr'.format(split): res[split]['MRR']}, step=step_idx)
                    wandb.log({'step_wise/{}/hits@1'.format(split): res[split]['hits@[1,3,10]'][0]}, step=step_idx)

                if res[split]['MRR'] > best_valid_mrr:
                    best_valid_mrr = res[split]['MRR']
                    self.model.checkpoint(model_cache_path=self.model_cache_path, epoch_id='best_valid')
                    if self.opt['cache_eval'] is not None:
                        for s in self.dataset.splits:
                            for m in ['lhs', 'rhs']:
                                torch.save(res_detailed[s][m], 
                                           self.opt['cache_eval']+'{s}_{m}.pt'.format(s=s, m=m))
                    wandb.run.summary['best_valid_mrr'] = best_valid_mrr
                    wandb.run.summary['best_valid_epoch'] = e
                    wandb.run.summary['corr_test_mrr'] = res['test']['MRR']
                    wandb.run.summary['corr_test_hits@1'] = res['test']['hits@[1,3,10]'][0]
                    wandb.run.summary['corr_test_hits@3'] = res['test']['hits@[1,3,10]'][1]
                    wandb.run.summary['corr_test_hits@10'] = res['test']['hits@[1,3,10]'][2]
            if best_valid_mrr == 1:
                print('MRR 1, diverged!')
                break
            if best_valid_mrr > 0 and best_valid_mrr < 2e-4:
                if l_reg_raw.item() < 1e-4:
                    print('0 embedding weight, diverged!')
                    break
        self.model.eval()
        results = self.dataset.eval(self.model, 'test', -1)
        print("\n\nTEST : ", results)
        wandb.run.summary['is_done'] = True