import random, time, json
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
#from ddpg_agent import Agent 
from ddpg_multi_agent import DemoAgent

# = = = = = = = = = Enviroment initialization = = = = = = = = = # 
env = UnityEnvironment(file_name='Reacher_Linux_multiple/Reacher.x86')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1] 

with open("hyperparameters.json", 'r') as f:
    settings = json.load(f)

episodes = 10
scores_deque = deque(maxlen=100)
scores = []
avg_scores = []
max_score = -np.Inf 

agent = DemoAgent(state_size, action_size, 0, 'success_checkpoint_actor.pth', 'success_checkpoint_critic.pth', settings)

for ith_episode in range(episodes):

    env_info = env.reset(train_mode=False)[brain_name]
    agents_scores = np.zeros(num_agents) 
    states = env_info.vector_observations 
    currentTimesteps = 0

    for timestep in range(1000): 

        agents_actions = np.zeros((num_agents, action_size), dtype=np.float32)
        for i, state in enumerate(states):
            agents_actions[i] = agent.act(state) 


        env_info    = env.step(agents_actions) 
        next_states = env_info[brain_name].vector_observations
        rewards     = env_info[brain_name].rewards
        dones       = env_info[brain_name].local_done
        

        states = next_states 
        agents_scores += rewards
        currentTimesteps = timestep
        if True in dones:
            break

        
        avg_score = np.mean(agents_scores) 
        scores_deque.append(avg_score) 
        scores.append(avg_score) 
        agent.current_avg_score = np.mean(scores_deque)    
        
    print("Episode: {}\t Average score: {}\t Score: {}".format(ith_episode, np.mean(scores_deque), avg_score))  

env.close()