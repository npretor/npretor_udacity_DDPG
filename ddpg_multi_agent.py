import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DemoAgent:
    def __init__(self, state_size, action_size, seed, actor_path, critic_path, settings):
        random.seed(seed)
        self.actor= Actor(state_size, action_size, random.random(), settings['actor_network_shape']).to(device)
        self.critic = Critic(state_size, action_size, random.random(), settings['critic_network_shape']).to(device)
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) 
        self.actor.eval()
        self.critic.eval() 
        with torch.no_grad():
            action_values = self.actor(state) 
        return action_values.cpu().data.numpy() 


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, settings, num_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.settings = settings
        self.num_agents = num_agents 

        # Let's set this in proximity to the goal. I could set a linear value, but that's not very helpful
        # Rather, if I set the noise multiplier amount proportional the proximity to a stated goal, so it falls off as I approach 
        # This method only works because I know I want to approach a score of 30 
        self.current_avg_score = 0.001 
        self.goal_avg_score = 30;
        
        # This should be zero or close to it at 30 or above, and approaching one as the current_avg approaches zero 
        if self.current_avg_score > 30:
            current_avg_score = 30.0
        else:
            self.current_avg_score = self.current_avg_score 
        self.noise_decay_rate = 1 - (self.current_avg_score/self.goal_avg_score) 

        self.actor_loss = 0.0 
        self.critic_loss = 0.0 

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, settings["actor_network_shape"]).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, settings["actor_network_shape"]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.settings["LR_ACTOR"])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, settings["critic_network_shape"]).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, settings["critic_network_shape"]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.settings["LR_CRITIC"], weight_decay=self.settings["WEIGHT_DECAY"])

        # Noise process
        self.noise = OUNoise(action_size * num_agents, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.settings["BUFFER_SIZE"], self.settings["BATCH_SIZE"], random_seed)
    
    def step(self, states, actions, rewards, next_states, dones, timestep):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        
        """
        # Save each experience 
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done) 

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.settings["BATCH_SIZE"] and timestep % self.settings["LEARN_EVERY"] == 0:
            experiences = self.memory.sample()
            self.learn(experiences, self.settings["GAMMA"]) 

    def act(self, state, add_noise=True):
        """
        states(list): 
            list of states for each agent 
        Returns: 
            actions for given state as per current policy 
        """
        # Convert states from numpy to tensor 
        state = torch.from_numpy(state).float().to(device) 
        
        # Set network to eval mode (as opposed to training mode)
        self.actor_local.eval() 

        # Get a state from actor_local and add it to the list of states for each actor  
        with torch.no_grad():
            #actions[i] = self.actor_local(states[i]).cpu().data.numpy() 
            action = self.actor_local(state).cpu().data.numpy() 

        self.actor_local.train()
        if add_noise:
            self.noise_decay_rate = 1 - (self.current_avg_score/self.goal_avg_score)
            action += (self.noise.sample().reshape((-1, 4)) * self.noise_decay_rate)   
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states) 
        Q_targets_next = self.critic_target(next_states, actions_next) 

        # Compute Q targets for current states (y_i) 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) 

        # Compute critic loss
        Q_expected = self.critic_local(states, actions) 
        critic_loss = F.mse_loss(Q_expected, Q_targets) 
        self.critic_loss = critic_loss

        # Minimize the loss
        self.critic_optimizer.zero_grad() 
        critic_loss.backward() 

        # Taken from: https://github.com/adaptationio/DDPG-Continuous-Control/blob/master/agent.py
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)     
        self.critic_optimizer.step() 

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_loss = actor_loss 

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.settings["TAU"]) 
        self.soft_update(self.actor_local, self.actor_target, self.settings["TAU"]) 

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2): 
        """Initialize parameters and noise process.""" 
        self.mu = mu * np.ones(size) 
        self.theta = theta 
        self.sigma = sigma 
        self.seed = random.seed(seed) 
        self.reset() 

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add batch of new experiences to memory."""
        # for i in range(len(states)):
        #     print("adding experience batch: {}".format(i))
        #     print(states[i])
        #     e = self.experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
        #     self.memory.append(e)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)