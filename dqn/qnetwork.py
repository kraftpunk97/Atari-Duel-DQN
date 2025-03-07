import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, num_actions: int, lr: float = 1e-5, *, device: str = "cpu"):
        super(QNetwork, self).__init__()  # Calling the superclass constructor
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=(8, 8),
                               stride=4)
        std_dev = math.sqrt(2.0 / (4 * 84 * 84))
        nn.init.normal_(self.conv1.weight, mean=0.0, std=std_dev)
        self.conv1.bias.data.fill_(0.0)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(4, 4),
                               stride=2)
        std_dev = math.sqrt(2.0 / (32 * 4 * 8 * 8))
        nn.init.normal_(self.conv2.weight, mean=0.0, std=std_dev)
        self.conv2.bias.data.fill_(0.0)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=1)
        std_dev = math.sqrt(2.0 / (64 * 32 * 4 * 4))
        nn.init.normal_(self.conv3.weight, mean=0.0, std=std_dev)
        self.conv3.bias.data.fill_(0.0)

        self.fc = nn.Linear(in_features=64*7*7, out_features=512)
        std_dev = math.sqrt(2.0 / (64 * 64 * 3 * 3))
        nn.init.normal_(self.fc.weight, mean=0.0, std=std_dev)
        self.fc.bias.data.fill_(0.0)

        self.V = nn.Linear(in_features=512, out_features=1) 
        self.A = nn.Linear(in_features=512, out_features=num_actions)

        self.get_qvalues = self.forward
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor):
        if len(state.shape) < 4:  # Adding a dimension if a single state observation, and not a mini-batch
            state = state.unsqueeze(dim=0)
        state = state.to(self.device)  # Tensor.to() is not an in-place operation
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        v = self.V(x)
        a = self.A(x)
        qvalues = v + (a - a.mean(dim=1, keepdim=True))
        return qvalues  # The output is a tensor with the Q-values of different actions for that state.

    def get_action(self, state: torch.Tensor):
        q_values = self.get_qvalues(state)
        action = torch.argmax(q_values).item()
        return action

    def train_step(self, masked_qvals: torch.Tensor, target_qvals: torch.Tensor):
        masked_qvals = masked_qvals.to(self.device)
        target_qvals = target_qvals.to(self.device)

        loss_func = nn.SmoothL1Loss()
        loss = loss_func(masked_qvals, target_qvals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
