import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------------------------------------
# Class Title: QNetwork 
#-----------------------------------------------------------------------
class QNetwork(nn.Module):
  """
  Deep Q-Network architecture.
  """
  #-----------------------------------------------------------------------
  # Function Title: __init__() 
  #-----------------------------------------------------------------------
  def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
    """
    Initialize parameters and build model.

    Params
    ======
    state_size (int): Dimension of each state
    action_size (int): Dimension of each action
    seed (int): Random seed
    fc1_units (int): Number of nodes in first hidden layer
    fc2_units (int): Number of nodes in second hidden layer
    """
    super().__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, fc1_units)
    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.out = nn.Linear(fc2_units, action_size)

  #-----------------------------------------------------------------------
  # Function Title: forward() 
  #-----------------------------------------------------------------------
  def forward(self, state):
    """
    Perform a forward pass where the network maps the state to action values.

    Params
    ======
    state (array) - list of states to convert
    """
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    return self.out(x)