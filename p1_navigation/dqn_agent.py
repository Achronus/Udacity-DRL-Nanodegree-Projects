import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------
# Class Title: Agent 
#-----------------------------------------------------------------------
class Agent():
  """
  An agent that uses Q-Learning to interact with and learn from an environment.
  """
  #-----------------------------------------------------------------------
  # Function Title: __init__() 
  #-----------------------------------------------------------------------
  def __init__(self, state_size, action_size, seed):
    """
    Initialize all Agent parameters.

    Params
    ======
    state_size (int): dimension of each state
    action_size (int): dimension of each action
    seed (int): random seed
    """
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)

    # Initialize Q-Networks and optimizer
    self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
    self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

    # Initialize replay buffer
    self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # Initialize time step
    self.t_step = 0

  #-----------------------------------------------------------------------
  # Function Title: step() 
  #-----------------------------------------------------------------------
  def step(self, state, action, reward, next_state, done):
    """
    Take a step through the environment.

    Params
    ======
    state (tuple): current state
    action (int): current action
    reward (float): current reward
    next_state (int): next state
    done (boolean): complete flag
    """
    # Save experience in replay memory
    self.memory.add(state, action, reward, next_state, done)

    # Learn every set of timesteps
    self.t_step = (self.t_step + 1) % UPDATE_EVERY

    if self.t_step == 0:
      # Check that enough samples are available in memory
      if len(self.memory) > BATCH_SIZE:
        # Get random set of samples and learn from set amount of experiences
        experiences = self.memory.sample()
        self.learn(experiences, GAMMA)

  #-----------------------------------------------------------------------
  # Function Title: act() 
  #-----------------------------------------------------------------------
  def act(self, state, eps=0.):
    """
    Returns actions for given state as per current policy.

    Params
    ======
    state (array): current state
    eps (float): epsilon, for epsilon-greedy action selection
    """
    # Initialize state to tensor
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    # Set main network to evaluation mode
    self.qnetwork_local.eval()

    # Disable gradients and calculate action-values
    with torch.no_grad():
      action_values = self.qnetwork_local(state)

      # Set main network back to training mode
      self.qnetwork_local.train()

    # Epsilon-greedy action selection
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  #-----------------------------------------------------------------------
  # Function Title: learn() 
  #-----------------------------------------------------------------------
  def learn(self, experiences, gamma):
    """
    Update value parameters using given batch of experience tuples.

    Params
    ======
    experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
    gamma (float): discount factor
    """
    # Set each experience (observation) component
    states, actions, rewards, next_states, dones = experiences

    # Get max predicted Q values (for next states) from target model
    Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

    # Compute Q targets for current states 
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

    # Get expected Q values from local model
    Q_expected = self.qnetwork_local(states).gather(1, actions)

    # Compute loss (Mean Squared Error)
    loss = F.mse_loss(Q_expected, Q_targets)

    # Minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Update target network
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

  #-----------------------------------------------------------------------
  # Function Title: soft_update() 
  #-----------------------------------------------------------------------
  def soft_update(self, local_model, target_model, tau):
    """
    Soft updates model parameters using Fixed Q-Targets.
    θ_target = τ * θ_local + (1 - τ) * θ_target

    Params
    ======
    local_model (PyTorch model): weights will be copied from (θ_local)
    target_model (PyTorch model): weights will be copied to (θ_target)
    tau (float): interpolation parameter 
    """
    # Iterate over each parameter in both the target and local networks
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      # Perform soft update
      target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

#-----------------------------------------------------------------------
# Class Title: ReplayBuffer 
#-----------------------------------------------------------------------
class ReplayBuffer:
  """
  Fixed-size buffer to store experience tuples.
  """
  #-----------------------------------------------------------------------
  # Function Title: __init__() 
  #-----------------------------------------------------------------------
  def __init__(self, action_size, buffer_size, batch_size, seed):
    """
    Initialize the ReplayBuffer parameters.

    Params
    ======
    action_size (int): dimension of each action
    buffer_size (int): maximum size of buffer
    batch_size (int): size of each training batch
    seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)  
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)

  #-----------------------------------------------------------------------
  # Function Title: add() 
  #-----------------------------------------------------------------------
  def add(self, state, action, reward, next_state, done):
    """
    Add a new experience to memory.
    """
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)

  #-----------------------------------------------------------------------
  # Function Title: sample() 
  #-----------------------------------------------------------------------
  def sample(self):
    """
    Randomly sample a batch of experiences from memory.
    """
    # Retrieve a random sample of experiences
    experiences = random.sample(self.memory, k=self.batch_size)

    # Store each experience component within its respective torch tensor
    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

    # Output the sampled torch tensors
    return (states, actions, rewards, next_states, dones) 

  #-----------------------------------------------------------------------
  # Function Title: __len__() 
  #-----------------------------------------------------------------------
  def __len__(self):
    """
    Return the current size of internal memory.
    """
    return len(self.memory)