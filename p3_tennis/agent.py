import numpy as np
import random

from model import Actor, Critic
from helper import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
  """Interacts with and learns from the environment, using a DDPG model.
  
  Parameters:
    - state_size (int): size of state space
    - action_size (int): size of action space
    - seed (int): Random seed
    - n_agents (int): number of agents
  """
  def __init__(self, state_size, action_size, seed, n_agents):
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)
    self.n_agents = n_agents

    # Actor Network (w/ Target Network)
    self.actor_local = Actor(state_size, action_size, seed).to(device)
    self.actor_target = Actor(state_size, action_size, seed).to(device)
    self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

    # Critic Network (w/ Target Network)
    critic_action_size = self.n_agents * self.action_size
    self.critic_local = Critic(state_size, critic_action_size, seed).to(device)
    self.critic_target = Critic(state_size, critic_action_size, seed).to(device)
    self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

    # Noise process for each agent
    self.noise = OUNoise(action_size, seed)
    self.noise_decay = 1

    # Ensure target weights are same as local
    self.soft_update_all(1)
  
  def act(self, state, add_noise=True):
    """Returns the actions for a given state, based on the current policy."""
    state = torch.from_numpy(state).float().to(device)
    self.actor_local.eval()

    with torch.no_grad():
      action = self.actor_local(state).cpu().data.numpy()

    self.actor_local.train()

    # Add noise to the action to explore the environment
    if add_noise:
      action += self.noise_decay * self.noise.sample()
    
    return np.clip(action, -1, 1)

  def reset(self):
    self.noise.reset()
  
  def soft_update_all(self, tau):
    """Soft updates both the actor and critic networks."""
    self.soft_update(self.critic_local, self.critic_target, tau)
    self.soft_update(self.actor_local, self.actor_target, tau)

  def soft_update(self, local_net, target_net, tau):
    """
    Soft update model parameters.
    θ_target =  τ * θ_local + (1 - τ) * θ_target

    Parameters:
      - local_net: PyTorch model, weights are copied from
      - target_net: PyTorch model, weights are copied to
      tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
      target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)