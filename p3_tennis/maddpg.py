import numpy as np
import random

from agent import Agent
from helper import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

UPDATE_EVERY = 2       # num timesteps before each update
NUM_UPDATE = 10        # num of updates after set num of timesteps
NOISE_DECAY = 0.9995   # reduce exploration slowly overtime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
  """MADDPG agent: interacts with and learns from the environment.
  
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

    # Set combined Replay Memory Buffer for all agents
    self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, n_agents, seed)

    # Initialize all agents
    self.agents = [Agent(self.state_size, self.action_size, seed, n_agents) for _ in range(self.n_agents)]

  def reset_agents(self):
    """Reset agents to default settings."""
    for agent in self.agents:
      agent.reset()

  def step(self, state, action, reward, next_state, done, timestep):
    """Saves each agents experience in replay memory and uses a random sample from the buffer to learn."""
    # Store experience
    self.memory.add(state, action, reward, next_state, done)

    # Learn, if there are enough samples available in memory
    if len(self.memory) > BATCH_SIZE and timestep % UPDATE_EVERY == 0:
      for _ in range(NUM_UPDATE):
        for agent_id in range(self.n_agents):
          experiences = self.memory.sample()
          self.learn(experiences, GAMMA, agent_id)

  def act(self, states, add_noise=True):
    """Returns the actions for a given state, based on the current policy."""
    actions = []
    for i, agent in enumerate(self.agents):
      action = agent.act(states[i:i+1], add_noise)
      actions.append(action)
    
    actions = np.concatenate(actions)
    return actions
  
  def learn(self, experiences, gamma, agent_id):
    """
    Restructures the given experiences and trains each agent based on those experiences.

    Parameters:
      - experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
      - gamma (float): discount factor
      - agent_id (int): agent id number
    """
    states, actions, rewards, next_states, dones = experiences

    state = states[agent_id]
    reward = rewards[agent_id]
    next_state = next_states[agent_id]
    done = dones[agent_id]

    # ----------- Update critic ----------- #
    # Get the predicted next-state actions and Q values from target models
    actions_next = [agent.actor_target(states[i]).detach() for i, agent in enumerate(self.agents)]
    actions_next = torch.cat(actions_next, dim=1)
    Q_targets_next = self.agents[agent_id].critic_target(next_state, actions_next)

    # Compute Q-targets for current states
    Q_targets = reward + (gamma * Q_targets_next * (1 - done))

    # Compute critic loss
    actions_expected = torch.cat(actions, dim=1)
    Q_expected = self.agents[agent_id].critic_local(state, actions_expected) # forward pass
    critic_loss = F.mse_loss(Q_expected, Q_targets)

    # Minimize loss
    self.agents[agent_id].critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.agents[agent_id].critic_local.parameters(), 1)
    self.agents[agent_id].critic_optimizer.step()

    # ----------- Update actor ----------- #
    # Compute actor loss
    actions_pred = [agent.actor_local(state) if i == agent_id else agent.actor_local(state).detach() for i, agent in enumerate(self.agents)]
    actions_pred = torch.cat(actions_pred, dim=1)
    actor_loss = -self.agents[agent_id].critic_local(state, actions_pred).mean()

    # Minimze loss
    self.agents[agent_id].actor_optimizer.zero_grad()
    actor_loss.backward()
    self.agents[agent_id].actor_optimizer.step()

    # -------- Reduce noise -------- #
    for agent in self.agents:
      agent.noise_decay *= NOISE_DECAY

    # -------- Update target networks -------- #
    self.soft_update_all()

  def soft_update_all(self):
    """Soft update the actor and critic networks for each agent."""
    for agent in self.agents:
      agent.soft_update_all(TAU)

  def save_model(self):
    """Save the model's parameters of each agent."""
    for idx, agent in enumerate(self.agents):
      torch.save(agent.actor_local.state_dict(), f'agent_{idx}_actor_checkpoint.pth')
      torch.save(agent.critic_local.state_dict(), f'agent_{idx}_critic_checkpoint.pth')

  def load_model(self, device):
    """Load the model's parameters of each agent."""
    for idx, agent in enumerate(self.agents):
      agent.actor_local.load_state_dict(torch.load(f'agent_{idx}_actor_checkpoint.pth', map_location=device))
      agent.critic_local.load_state_dict(torch.load(f'agent_{idx}_critic_checkpoint.pth', map_location=device))