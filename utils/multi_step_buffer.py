from collections import deque, namedtuple
from utils.buffer_deque import CustomDeque
import random
import torch


class multi_step_ReplayBuffer:
    def __init__(self, capacity, N_STEP=3):
        self.N_STEP = N_STEP
        self.memory = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=N_STEP)
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        #self.state_dim = state_dim

    def push(self, *args):
        self.n_step_buffer.append(self.transition(*args))
        if len(self.n_step_buffer) == self.N_STEP:
            reward, next_state = self.get_n_step_info()
            state, action = self.n_step_buffer[0].state, self.n_step_buffer[0].action
            self.memory.append(self.transition(state, action, reward, next_state))

    def get_n_step_info(self, GAMMA=0.9):
        reward, next_state = self.n_step_buffer[-1].reward, self.n_step_buffer[-1].next_state
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            reward = transition.reward + GAMMA * reward
            next_state = transition.next_state
        return reward, next_state

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = self.transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to('cuda')
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to('cuda')
        next_state_batch = torch.cat(batch.next_state)

        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.memory)