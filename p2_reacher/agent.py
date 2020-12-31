import numpy as np
import random

from model import Actor, Critic
from helper import OUNoise, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
EPS_START = 1           # epsilon start value
EPS_END = 0.05          # epsilon ending value
EPS_DECAY = 1e-6        # epsilon decay rate
UPDATE_EVERY = 20       # num timesteps before each update
NUM_UPDATE = 10         # num of updates after set num of timesteps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
  """Interacts with and learns from the environment.
  
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
    self.eps = EPS_START

    # Actor Network (w/ Target Network)
    self.actor_local = Actor(state_size, action_size, seed).to(device)
    self.actor_target = Actor(state_size, action_size, seed).to(device)
    self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

    # Critic Network (w/ Target Network)
    self.critic_local = Critic(state_size, action_size, seed).to(device)
    self.critic_target = Critic(state_size, action_size, seed).to(device)
    self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

    # Noise process for each agent
    self.noise = OUNoise((self.n_agents, action_size), seed)

    # Replay memory
    self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # Ensure target weights are same as local
    self.hard_update(self.actor_local, self.actor_target)
    self.hard_update(self.critic_local, self.critic_target)

  def step(self, states, actions, rewards, next_states, dones, timestep):
    """Saves each agents experience in replay memory and uses a random sample from the buffer to learn."""
    # Save experiences
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
      self.memory.add(state, action, reward, next_state, done)

    # Learn, if there are enough samples available in memory
    if len(self.memory) > BATCH_SIZE and timestep % UPDATE_EVERY == 0:
      for _ in range(NUM_UPDATE):
        experiences = self.memory.sample()
        self.learn(experiences, GAMMA)
  
  def act(self, state, add_noise=True):
    """Returns the actions for a given state, based on the current policy."""
    state = torch.from_numpy(state).float().to(device)
    self.actor_local.eval()

    with torch.no_grad():
      action = self.actor_local(state).cpu().data.numpy()

    self.actor_local.train()

    # Add noise to the action to explore the environment
    if add_noise:
      action += self.eps * self.noise.sample()
    
    return np.clip(action, -1, 1)

  def reset(self):
    self.noise.reset()

  def learn(self, experiences, gamma):
    """
    Updates the policy and value parameters using the given batch of experiences.
    Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
    where:
      actor_target(state) -> action
      critic_target(state, action) -> Q-value

    Parameters:
      - experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
      - gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experiences

    # ----------- Update critic ----------- #
    # Get the predicted next-state actions and Q values from target models
    actions_next = self.actor_target(next_states)
    Q_targets_next = self.critic_target(next_states, actions_next)

    # Compute Q-targets for current states
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

    # Compute critic loss
    Q_expected = self.critic_local(states, actions) # forward pass
    critic_loss = F.mse_loss(Q_expected, Q_targets)

    # Minimize loss
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
    self.critic_optimizer.step()

    # ----------- Update actor ----------- #
    # Compute actor loss
    actions_pred = self.actor_local(states) # forward pass
    actor_loss = -self.critic_local(states, actions_pred).mean()

    # Minimze loss
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # -------- Update target networks -------- #
    self.soft_update(self.critic_local, self.critic_target, TAU)
    self.soft_update(self.actor_local, self.actor_target, TAU)

    # -------- Update noise -------- #
    self.eps -= EPS_DECAY
    self.eps = max(EPS_END, self.eps)
    self.noise.reset()
  
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

  def hard_update(self, local, target):
    """
    Hard update model parameters.

    Parameters:
      - local: PyTorch model, weights are copied from
      - target: PyTorch model, weights are copied to
    """
    for target_param, local_param in zip(target.parameters(), local.parameters()):
      target_param.data.copy_(local_param.data)
