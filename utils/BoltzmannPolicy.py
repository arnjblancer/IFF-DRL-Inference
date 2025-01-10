import numpy as np
import torch
class BoltzmannPolicy:
    def __init__(self, tau=1.0):
        self.tau = tau

    def select_action(self, q_values):
        q_values = q_values.cpu().numpy()
        exp_values = np.exp(q_values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        action = np.random.choice(np.arange(len(probabilities)), p=probabilities)

        return torch.tensor([action], dtype=torch.int64).to('cuda')
