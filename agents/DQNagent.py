import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import math
from random import random
from utils.ReplayBuffer import ReplayBuffer
from utils.BoltzmannPolicy import BoltzmannPolicy
from trainers.utils import MSEWithL2Regularization
from Model.TimesNet import TimesNet



class DDQN:
    def __init__(self, cfg):
        self.cfg = cfg
        print('algorithm: ', cfg.agent.algorithm)
        print(cfg.agent.model)
        self.policy_net = eval(cfg.agent.model)(OmegaConf.load('configs/{}.yaml'.format(cfg.agent.model))).to(cfg.device)
        self.target_net = eval(cfg.agent.model)(OmegaConf.load('configs/{}.yaml'.format(cfg.agent.model))).to(cfg.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.train.lr)
        self.gamma = cfg.train.gamma
        self.buffer_size = cfg.agent.buffer_size
        self.memory = ReplayBuffer(self.buffer_size)
        # self.batch_size = cfg.train.batch_size
        self.target_update = cfg.train.target_update
        self.loss_function = MSEWithL2Regularization()
        self.num_updates = 0
        self.EPS_START = cfg.train.EPS_START
        self.EPS_END = cfg.train.EPS_END
        self.EPS_DECAY = cfg.train.EPS_DECAY
        self.steps_done = 0
        self.BoltzmannPolicy = BoltzmannPolicy()
        self.memory_counter = 0
        self.tau = 0.1
        self.epsilon = 0

    def epsilonGreedAction(self, q_values):

        epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random() > epsilon:
            return torch.argmax(q_values)
        else:
            return torch.tensor(np.random.randint(0, 3)).to(self.cfg.device)

    def act(self, state, train=True):

        state = state.to(self.cfg.device)
        with torch.no_grad():
            q_values = self.policy_net(state, None, None, None).squeeze()
            if train:
                action = self.epsilonGreedAction(q_values)
            else:
                action = torch.argmax(q_values)
        return action






    def learn(self, batch):
        self.policy_net.train()

        s, a, r, s_ = batch
        q_eval = self.policy_net(s, None, None, None).squeeze()  # batch_size,3
        with torch.no_grad():
            Q_B = self.target_net(s_, None, None, None).squeeze()
            Q_A = self.policy_net(s_, None, None, None).squeeze()
            _, Q_A_max_index = torch.max(Q_A, dim=1)
            q_target = r + self.gamma * Q_B[np.arange(s.shape[0], dtype=np.int32), Q_A_max_index]

        q_target = q_target.detach()
        q_eval = q_eval.gather(1, a.unsqueeze(1)).squeeze()

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q_eval, q_target)
        loss.backward()
        self.optimizer.step()

        self.num_updates += 1
        if self.num_updates % self.target_update == 0:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
