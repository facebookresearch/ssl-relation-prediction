# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch


class F2(object):
    def __init__(self, lmbda: float):
        super(F2, self).__init__()
        self.lmbda = lmbda

    def penalty(self, x, factors): #TODO: remove x
        norm, raw = 0, 0
        for f in factors:
            raw += torch.sum(f ** 2)
            norm += self.lmbda * torch.sum(f ** 2)
        return norm / factors[0].shape[0], raw / factors[0].shape[0], self.lmbda
    
    def checkpoint(self, regularizer_cache_path, epoch_id):
        if regularizer_cache_path is not None:
            print('Save the regularizer at epoch {}'.format(epoch_id))
            path = regularizer_cache_path + '{}.reg'.format(epoch_id)
            torch.save(self.state_dict(), path)
            print('Regularizer Checkpoint:{}'.format(path))

class N3(object):
    def __init__(self, lmbda: float):
        super(N3, self).__init__()
        self.lmbda = lmbda

    def penalty(self, x, factors):
        """

        :param factors: tuple, (s, p, o), batch_size * rank
        :return:
        """
        norm, raw = 0, 0
        for f in factors:
            raw += torch.sum(
                torch.abs(f) ** 3
            )
            norm += self.lmbda * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0], raw / factors[0].shape[0], self.lmbda
    
    def checkpoint(self, regularizer_cache_path, epoch_id):
        if regularizer_cache_path is not None:
            print('Save the regularizer at epoch {}'.format(epoch_id))
            path = regularizer_cache_path + '{}.reg'.format(epoch_id)
            torch.save(self.state_dict(), path)
            print('Regularizer Checkpoint:{}'.format(path))