import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from agent import Agent, ReplayBuffer

n_episodes = 100
from collections import deque

# Randomly initialize critic network and actor 
# Initialize target network Q and u
# Initialize replay buffer 

"""
Initialize the environment 
Agent gets environment state 
Agent infers an action from the state 
    Get state from network 
    Select action based on explore/exploit percentage 
Apply action to environment and get an experience => state, action, reward, done


Add experience to deque of experiences
Retrain every n experiences
"""


# = = = = = = = = Environment initialization = = = = = = = = # 
from unityagents import UnityEnvironment

env = UnityEnvironment(file_name='...') 
brain_name = env.brain_names[0] 
brain = env.brains[brain_name]  
env_info = env.reset(train_mode=True)[brain_name] 
num_agents = len(env_info.agents) 
action_size = brain.vector_action_space_size 
states = env_info.vector_observations 
state_size = states.shape[1] 
 
print('Number of agents:', num_agents)
print('Size of each action:', action_size)
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])



num_episodes = 100 
experiences = deque(maxlen=100) 
agent = Agent(5, 10) 
max_timesteps = 100 


action = agent.act() 
state, action, reward, done = env.receive_action(action) 
experiences.append([state, action, reward, done]) 


for episode_num in range(1, num_episodes):

    state = env_info.vector_observations[0]
    for timestep in range(max_timesteps):
        score = 0
        action = agent.act(state) 
        env_info = env.step(action)
        next_state, reward, done = env_info.vector_observations[0]
        agent.step(state, action, reward, next_state, done)
        state = next_state 
        score += reward 
        if done: 
            break 


class DQN_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQN_model, self).__init__() 
        self.fc1 = nn.Linear(input_size, hidden_size)   
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.fc3 = nn.Lienar(hidden_size, output_size)  


    def forward(self, state):
        x = nn.Relu(self.fc1(state))
        x = nn.Relu(self.fc2(x))
        action = nn.Relu(self.fc3(x))
        return action 


class Agent:
    def __init__(self, input_size, output_size):
        self.memory = ReplayBuffer(memory_size=100) 
        self.target_actor = DQN_model(input_size, output_size)  
        self.target_critic = DQN_model(input_size, output_size) 
        self.main_actor = DQN_model(input_size, output_size)  
        self.main_critic = DQN_model(input_size, output_size) 


    def act(self, state, exp_exp_rati0=0.5):
        """
        State: 
        Explore vs exploit ratio: 

        """
        if np.random.random > exp_exp_rati0:
            # select random action from the possible states 
            # Map from 0-1.0 to joint value distribution
            # 
            pass
        else:
            # Select the action using the target critic network
            return np.argmax(self.target_critic.forward(state)) 

        return action 


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done) 

        # Sample a batch of transitions 
        # Set: y_i = reward + gamma*( self.) 
        # Update critic 
        #    L = (1/n) * y_i - 

    def learn(self):

        pass 


class ReplayBuffer:
    def __init__(self, memory_size): 
        self.experiences = deque(maxlen=size) 


    def add(self, state, action, reward, next_state): 
        self.experiences.add([state, action, reward, next_state]) 