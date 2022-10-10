import torch 
import torch.nn as nn
import torch.nn.functional as F
from random import seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, seed=0) -> None:
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)

        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)

        self.replay_buffer = ReplayBuffer()

    def step(self):
        """
        For each step, save  
        """
        pass

    def act(self, state):
        """
        Actions come from the actor_local

        """
        chosen_action = self.actor_local(state)

        self.replay_buffer.add()

        self.learn()

    def learn(self):
        """
        Sample 
        
        """
        pass 

    def soft_update(self):
        pass 


class ReplayBuffer:
    def __init__(self) -> None:
        """
        Stores a window of experiences 
        """
        self.memory = [] 

    def add(self, state, action, reward, next_state, done): 
        experience = [state, action, reward, next_state, done] 
        self.memory.insert(0, experience) 
        self.memory.pop(-1) 


class Actor:
    def __init__(self, input_size, output_size, seed) -> None:
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

    def forward(self, state): 
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x 

class Critic:
    def __init__(self, input_size, output_size, seed) -> None:
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

    def forward(self, state): 
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x 