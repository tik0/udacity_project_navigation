import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of hidden units in the first layer
            fc2_units (int): Number of hidden units in the second layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Hyperparameters for the network
        self.input_size = state_size
        self.hidden_sizes = [fc1_units, fc2_units]
        self.output_size = action_size

        # The feed-forward network
        self.model = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(self.input_size, self.hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
                              ('relu2', nn.ReLU()),
                              ('logits', nn.Linear(self.hidden_sizes[1], self.output_size))]))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model.forward(state)
