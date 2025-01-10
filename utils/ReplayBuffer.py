from collections import deque
from utils.buffer_deque import CustomDeque
import random
import torch

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def clear(self):
        self.buffer.clear()

    def expert_add(self, state, action, reward, next_state):
        self.buffer.append((state, action.reshape(1, -1), reward.unsqueeze(0).reshape(1, -1), next_state))
    def Agent_add(self, state, action, reward, next_state):
        self.buffer.append((state, action.reshape(1, -1), reward.unsqueeze(0).reshape(1, -1), next_state))

    def Agent_sample(self, batch_size):
        states, actions, rewards, next_states = zip(*random.sample(self.buffer, batch_size))
        return torch.concat(states, dim=0), torch.concat(actions, dim=0).squeeze(1), torch.concat(rewards, dim=0).squeeze(1), torch.concat(next_states, dim=0)

    def expert_sample(self, batch_size):
        states, actions, rewards, next_states = zip(*random.sample(self.buffer, batch_size))
        return torch.concat(states, dim=0), torch.concat(actions, dim=0), torch.concat(rewards,dim=0), torch.concat(next_states, dim=0)

    def Agent_expert_add(self, state, action, reward, expert_action, next_state):
        self.buffer.append((state, action.reshape(1, -1), reward.unsqueeze(0).reshape(1, -1), expert_action.reshape(1, -1), next_state))

    def Agent_expert_sample(self, batch_size):
        states, actions, rewards, expert_actions, next_states = zip(*random.sample(self.buffer, batch_size))
        return torch.concat(states, dim=0), torch.concat(actions, dim=0).squeeze(1), torch.concat(rewards, dim=0).squeeze(1),torch.concat(expert_actions, dim=0).squeeze(1), torch.concat(next_states, dim=0)

    def PPO_Agent_add(self, state, action, reward, next_state, prob):
        self.buffer.append((state, action.reshape(1, -1), reward.unsqueeze(0).reshape(1, -1), next_state, prob.reshape(1,-1)))

    def PPO_Agent_sample(self, batch_size):
        states, actions, rewards, next_states, probs = zip(*random.sample(self.buffer, batch_size))
        return torch.concat(states, dim=0), torch.concat(actions, dim=0).squeeze(1), torch.concat(rewards, dim=0).squeeze(1), torch.concat(next_states, dim=0), torch.concat(probs, dim=0).squeeze(1)

    def __len__(self):
        return len(self.buffer)