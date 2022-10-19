import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from ddpg_agent import Agent 

# = = = = = = = = = Enviroment initialization = = = = = = = = = # 
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86')
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


agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10) 

def ddpg(num_episodes=1000, max_timesteps=500):
    scores_deque = deque(max_len=100)
    scores = []
    max_score = -np.Inf 

    for ith_episode in range(1, num_episodes+1): 
        state = env.reset()
        agent.reset() 
        score = 0 

        for timestep in range(max_timesteps): 
            action = agent.act(state) 

            next_state, reward, done, _ = env.step(action) 
            agent.step(state, action, reward, next_state, done) 

            state = next_state 
            score += reward 
            if done:
                break
        
        scores_deque.append(score) 
        scores.append(score) 

        print("Episode: {}\t Average score: {}\t Score: {}".format(ith_episode, np.mean(scores_deque), score)) 

        if ith_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")   
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth") 
            print("Episode: {}\t Average score: {}".format(ith_episode, np.mean(scores_deque))) 

    return scores 

scores = ddpg() 

fig = plt.figure() 
ax = fig.add_subplot(111) 
plt.plot(np.arange(1, len(scores)+1), scores) 
plt.ylabel('Score') 
plt.xlabel('Episode #') 
plt.show() 


