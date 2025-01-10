from torch.distributions import Categorical
import torch
from omegaconf import OmegaConf
from torch import optim
import torch.nn as nn
from utils.ReplayBuffer import ReplayBuffer
from Model.TimesNet import TimesNet
class PPO:
    def __init__(self, cfg):
        self.cfg = cfg
        print('algorithm: ', cfg.agent.algorithm)
        print(cfg.agent.model)
        self.policy_net = eval(cfg.agent.model)(OmegaConf.load('configs/{}.yaml'.format(cfg.agent.model)), 'agent').to(cfg.device)
        self.critic_net = eval(cfg.agent.model)(OmegaConf.load('configs/{}.yaml'.format(cfg.agent.model)), 'critic').to(cfg.device)
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.train.lr)
        self.crtic_optimizer = optim.Adam(self.critic_net.parameters(), lr=cfg.train.lr)
        self.gamma = cfg.train.gamma
        self.buffer_size = cfg.agent.buffer_size
        self.memory = ReplayBuffer(self.buffer_size)
        # self.batch_size = cfg.train.batch_size
        self.target_update = cfg.train.target_update
        self.num_updates = 0
        self.EPS_START = cfg.train.EPS_START
        self.EPS_END = cfg.train.EPS_END
        self.EPS_DECAY = cfg.train.EPS_DECAY
        self.steps_done = 0
        self.memory_counter = 0
        self.epsilon = 0
        self.policy_clip = 0.1





    def act(self, state,train=True):

        pi, v = self.policy_net(state, None, None, None), self.critic_net(state, None, None, None)
        probs = torch.softmax(pi, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action, log_probs

    def learn(self, batch):
        self.policy_net.train()
        self.critic_net.train()

        s, a, r, s_, old_log_probs = batch
        old_pi, old_v = self.policy_net(s, None, None, None), self.critic_net(s, None, None, None)
        old_v_ = self.critic_net(s_, None, None, None)
        advantage =r + self.gamma * old_v_.squeeze() - old_v.squeeze()

        new_pi, new_v = self.policy_net(s, None, None, None), self.critic_net(s, None, None, None)
        new_v_ = self.critic_net(s_, None, None, None)
        new_probs = torch.softmax(new_pi, dim=-1)
        dist = Categorical(new_probs)
        new_log_probs = dist.log_prob(a)

        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(new_v.squeeze(), r + self.gamma * old_v_.squeeze())
        total_loss = (actor_loss + 0.5 * critic_loss).mean()
        self.actor_optimizer.zero_grad()
        self.crtic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.crtic_optimizer.step()



