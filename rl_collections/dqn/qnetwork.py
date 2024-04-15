from collections import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, num_actions: int, lr: float = 0.001, *, device: str = "cpu"):
        super().__init__()  # Calling the superclass constructor
        self.device = device
        print(f'Device: {self.device}')
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=16,
                               kernel_size=(8, 8),
                               stride=4)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=(4, 4),
                               stride=2)
        self.fc = nn.Linear(in_features=2048, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=num_actions)
        self.get_qvalues = self.forward
        self.optimizer = optim.RMSprop(params=self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor):
        if len(state.shape) < 4:  # Adding a dimension if a single state observation, and not a mini-batch
            state = state.unsqueeze(dim=0)
        state = state.to(self.device)  # Tensor.to() is not an in-place operation
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        qvalues = F.relu(self.output(x))
        return qvalues  # The output is a tensor with the Q-values of different actions for that state.

    def get_action(self, state: torch.Tensor):
        q_values = self.get_qvalues(state)
        action = torch.argmax(q_values).item()
        return action

    def optimize(self, minibatch: tuple, y: torch.Tensor):
        state_minibatch, action_minibatch = minibatch
        state_minibatch = state_minibatch.to(self.device)  # Tensor.to() is not an in-place operation
        y = y.to(self.device)

        self.optimizer.zero_grad()

        qvalues = self.forward(state_minibatch)
        y_predicted = torch.tensor([qvalues[i, action] for (i, action) in enumerate(action_minibatch)],
                                   device=self.device, requires_grad=True)

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(y_predicted, y)

        loss.backward()
        self.optimizer.step()
        return loss
