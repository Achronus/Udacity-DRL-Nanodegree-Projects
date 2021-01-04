import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
  """Normalize weights to provide a faster convergence."""
  n_in = layer.weight.data.size()[0]
  limit = 1. / np.sqrt(n_in)
  return (-limit, limit)

class Actor(nn.Module):
  """
  Actor (Policy) Model.

  Parameters:
    - state_size (int): size of state space
    - action_size (int): size of action space
    - seed (int): Random seed
    - fc1_size (int): first hidden layer size
    - fc2_size (int): second hidden layer size
  """
  def __init__(self, state_size, action_size, seed=0, fc1_size=256, fc2_size=128):
    super().__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, fc1_size)
    self.fc2 = nn.Linear(fc1_size, fc2_size)
    self.out = nn.Linear(fc2_size, action_size)
    self.bn1 = nn.BatchNorm1d(fc1_size)
    self.initialize_weights()

  def initialize_weights(self):
    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    self.out.weight.data.uniform_(-3e-3, 3e-3)
  
  def forward(self, state):
    """Use the actor to map state -> actions."""
    x = F.relu(self.bn1(self.fc1(state)))
    x = F.relu(self.fc2(x))
    return F.tanh(self.out(x))

class Critic(nn.Module):
  """
  Critic (Value) Model.
  
  Parameters:
    - state_size (int): size of state space
    - action_size (int): size of action space
    - seed (int): Random seed
    - fc1_size (int): first hidden layer size
    - fc2_size (int): second hidden layer size
  """
  def __init__(self, state_size, action_size, seed=0, fc1_size=256, fc2_size=128):
    super().__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, fc1_size)
    self.fc2 = nn.Linear(fc1_size + action_size, fc2_size)
    self.out = nn.Linear(fc2_size, 1)
    self.bn1 = nn.BatchNorm1d(fc1_size)
    self.initialize_weights()

  def initialize_weights(self):
    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    self.out.weight.data.uniform_(-3e-3, 3e-3)
  
  def forward(self, state, action):
    """Use the critic to map (state, action) pairs -> Q-values."""
    xs = F.relu(self.bn1(self.fc1(state)))
    x = torch.cat((xs, action), dim=1)
    x = F.relu(self.fc2(x))
    return self.out(x)