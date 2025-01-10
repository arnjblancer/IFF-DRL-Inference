import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from trainers.utils import getBatch

# Time Series Prediction Model
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Online EWC implementation
class OnlineEWC:
    def __init__(self,netcfg, model, dataloader, importance=1000, gamma=1.0):
        self.netcfg = netcfg
        self.model = model
        self.importance = importance
        self.gamma = gamma
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._init_precision_matrices()
        self._update_means()
        self._first_precision_matrices(dataloader)

    def _init_precision_matrices(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)
        return precision_matrices

    def _first_precision_matrices(self, dataloader):
        precision_matrices = self._init_precision_matrices()

        self.model.eval()
        for iteration, batch in enumerate(dataloader):
            _,state, trend, mask, dayValue, weekValue = getBatch(batch)
            state = state.to('cuda')
            trend = trend.to('cuda')
            mask = mask.to('cuda')
            self.model.zero_grad()
            if self.netcfg.AutoCon:
                output,_ = self.model(state, mask, None, None)
            else:
                output = self.model(state, mask, None, None)
            loss = F.mse_loss(output, trend)
            loss.backward()

            for n, p in self.params.items():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(dataloader)

        # Combine old and new Fisher information matrices
        for n in precision_matrices:
            self._precision_matrices[n] = self.gamma * self._precision_matrices[n] + precision_matrices[n]

    def _update_means(self):
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _update_precision_matrices(self, dataloader):
        precision_matrices = self._init_precision_matrices()

        self.model.eval()

        state, trend, mask = dataloader
        self.model.zero_grad()
        if self.netcfg.AutoCon:
            output, _ = self.model(state, mask, None, None)
        else:
            output = self.model(state, mask, None, None)
        output = output.squeeze()
        loss = F.mse_loss(output, trend)
        loss.backward()

        for n, p in self.params.items():
            if p.grad is not None:
                precision_matrices[n].data += p.grad.data ** 2 / len(dataloader)

        for n in precision_matrices:
            self._precision_matrices[n] = self.gamma * self._precision_matrices[n] + precision_matrices[n]


    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss




