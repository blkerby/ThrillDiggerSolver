import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, size_x, size_y, hidden_widths):
        super().__init__()
        layers = []
        input_width = size_x * size_y * 8
        widths = [input_width] + hidden_widths
        for i in range(len(widths) - 1):
            layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
            layers.append(torch.nn.ReLU())
        self.sequential = torch.nn.Sequential(*layers)

        self.state_value_lin = torch.nn.Linear(widths[-1], 1)

        num_actions = size_x * size_y
        self.action_advantage_lin = torch.nn.Linear(widths[-1], num_actions)

    def forward(self, X):
        X = F.one_hot(X.to(torch.long), num_classes=8).to(torch.float32).view(X.shape[0], -1)
        X = self.sequential(X)
        state_value = self.state_value_lin(X)[:, 0]
        action_advantage = self.action_advantage_lin(X)
        action_value = action_advantage + state_value.unsqueeze(1)
        return state_value, action_value
