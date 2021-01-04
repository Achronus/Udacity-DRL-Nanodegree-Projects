import numpy as np
import copy
import random
from collections import namedtuple, deque

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
  """Ornstein-Uhlenbeck process."""
  def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
    self.size = size
    self.mu = mu * np.ones(size)
    self.theta = theta
    self.sigma = sigma
    self.seed = random.seed(seed)
    self.reset()

  def reset(self):
    """Reset the internal state (noise) to mean (mu)."""
    self.state = copy.copy(self.mu)
  
  def sample(self):
    """Update internal state and return it as a noise sample."""
    x = self.state
    x_delta = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
    self.state = x + x_delta
    return self.state

class ReplayBuffer:
  """
  Fixed-size buffer to store experiences.
  
  Parameters:
    - buffer_size (int): maximum size of buffer
    - batch_size (int): size of each training batch
    - n_agents (int): number of agents
    - seed (int): random seed number
  """
  def __init__(self, buffer_size, batch_size, n_agents, seed=0):
    self.memory = deque(maxlen=buffer_size) # Internal memory
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.n_agents = n_agents
    self.seed = random.seed(seed)

  def add(self, states, actions, rewards, next_states, dones):
    """Add a new experience to memory."""
    exp = self.experience(states, actions, rewards, next_states, dones)
    self.memory.append(exp)

  def sample(self):
    """Randomly sample a batch of experiences from memory for each agent."""
    experiences = random.sample(self.memory, k=self.batch_size)
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for i in range(self.n_agents):
      states.append(torch.from_numpy(np.vstack([e.state[i] for e in experiences if e is not None])).float().to(device))
      actions.append(torch.from_numpy(np.vstack([e.action[i] for e in experiences if e is not None])).float().to(device))
      rewards.append(torch.from_numpy(np.vstack([e.reward[i] for e in experiences if e is not None])).float().to(device))
      next_states.append(torch.from_numpy(np.vstack([e.next_state[i] for e in experiences if e is not None])).float().to(device))
      dones.append(torch.from_numpy(np.vstack([e.done[i] for e in experiences if e is not None]).astype(np.uint8)).float().to(device))

    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)