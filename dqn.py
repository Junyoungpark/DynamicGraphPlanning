from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

# from env import format_data
import pdb

class DQNTrainer(TorchTrainer):
    def __init__(self, qf, target_qf, learning_rate=1e-3, soft_target_tau=1e-3, target_update_period=1, 
                 qf_criterion=None, discount=0.99, reward_scale=1.0, args=[], format_data=None):
        super().__init__()
        self.qf = qf
        self.target_qf = target_qf
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=self.learning_rate)
        
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.args = args
        
        if format_data == None:
            self.format_fn = lambda x : x
        else:
            self.format_fn = format_data

    def train_from_torch(self, batch):
        rewards = torch.Tensor(batch['rewards']).unsqueeze(-1) * self.reward_scale
        terminals = torch.Tensor(batch['terminals'])
        actions = torch.Tensor(batch['actions'])

        obs = batch['observations']
        next_obs = batch['next_observations']

        """
        Compute loss
        """        
        ff = lambda x: self.target_qf(*x, *self.args)
        out = torch.stack(list(map(ff, map(self.format_fn, obs))), axis=0).cpu()
        
        target_q_values = out.max(-1, keepdims=True)[0].sum(1)        
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        
        ff = lambda x: self.qf(*x, *self.args)
        out = torch.stack(list(map(ff, map(self.format_fn, obs))), axis=0).cpu()
        
        actions_one_hot = F.one_hot(actions.to(torch.int64))
        y_pred = torch.sum(out * actions_one_hot, dim=-1).sum(1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)
        
#         pdb.set_trace()

        """
        Soft target network updates
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))
            print('qf loss:', self.eval_statistics['QF Loss'])
            print('total reward:', rewards.sum().item(), '\n')
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
        ]

    def get_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
        )
