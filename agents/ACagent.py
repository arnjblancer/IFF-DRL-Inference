from torch.distributions import Categorical
import torch
from omegaconf import OmegaConf
from torch import optim
import torch.nn as nn
from utils.ReplayBuffer import ReplayBuffer
from Model.TimesNet import TimesNet
class A2C:
    def __init__(self, cfg):
        self.cfg = cfg
        print('algorithm: ', cfg.agent.algorithm)
        print(cfg.agent.model)
        self.policy_net = eval(cfg.agent.model)(OmegaConf.load('configs/{}.yaml'.format(cfg.agent.model)), 'agent').to(cfg.device)
        self.critic_net = eval(cfg.agent.model)(OmegaConf.load('configs/{}.yaml'.format(cfg.agent.model)), 'critic').to(cfg.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.train.lr)
        self.gamma = cfg.train.gamma
        self.buffer_size = cfg.agent.buffer_size
        self.memory = ReplayBuffer(self.buffer_size)
        self.target_update = cfg.train.target_update
        self.num_updates = 0
        self.EPS_START = cfg.train.EPS_START
        self.EPS_END = cfg.train.EPS_END
        self.EPS_DECAY = cfg.train.EPS_DECAY
        self.steps_done = 0
        self.memory_counter = 0
        self.epsilon = 0

    def act(self, state,train=True):
        pi, v = self.policy_net(state, None, None, None), self.critic_net(state, None, None, None)
        probs = torch.softmax(pi, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action

    def learn(self, batch):
        self.policy_net.train()
        self.critic_net.train()

        s, a, r, s_ = batch
        pi, v = self.policy_net(s, None, None, None), self.critic_net(s, None, None, None) # batch_size,3 !!!!!!!!!!!!!!!!!
        v_ = self.critic_net(s_, None, None, None).detach()
        target_return = v_.squeeze() + r
        critic_loss = nn.MSELoss()(v.squeeze(), target_return)

        probs = torch.softmax(pi, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(a)
        actor_loss = -log_probs * (target_return - v.squeeze())
        total_loss = (actor_loss + critic_loss).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

