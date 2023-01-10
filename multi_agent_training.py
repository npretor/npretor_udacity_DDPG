import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
#from ddpg_agent import Agent 
from ddpg_multi_agent import Agent
import wandb

# Hyperparameters
settings = {
    "BUFFER_SIZE": int(100000),     # replay buffer size 
    "BATCH_SIZE" : 256,             # minibatch size 
    "GAMMA" : 0.99,                  # discount factor 
    "TAU" : 1e-3,                    # for soft update of target parameters 
    "LR_ACTOR" : 1e-3,               # learning rate of the actor   
    "LR_CRITIC" : 1e-3,              # learning rate of the critic  
    "WEIGHT_DECAY": 0,              # L2 weight decay
    "num_episodes": 1000, 
    "max_timesteps": 1000, 
    "actor_network_shape": [256, 128, 0],
    "critic_network_shape": [256, 128, 0], 
    "LEARN_EVERY": 0
} 


wandb.init(project="npretor_udacity_DDPG-multiagent", config=settings) 


# = = = = = = = = = Enviroment initialization = = = = = = = = = # 
env = UnityEnvironment(file_name='Reacher_Linux_multiple/Reacher.x86')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1] 



agent_num = 0
print('Number of agents:    ', num_agents) 
print('Size of each action: ', action_size) 
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size)) 
print('The state for the first agent looks like:  ', states[agent_num]) 
print("Rewards:   ",env_info.rewards[agent_num]) 
print("Observations:   ",env_info.vector_observations[agent_num] ) 
print("Done status:    ",env_info.local_done[agent_num]) 

agent = Agent(state_size=len(states[0]), action_size=action_size, random_seed=10, settings=settings, num_agents=num_agents) 

def ddpg(num_episodes, max_timesteps=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    avg_scores = []
    max_score = -np.Inf 
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    
    for ith_episode in range(1, num_episodes+1): 
        agent.reset() 
        agents_scores = np.zeros(num_agents) 

        for timestep in range(max_timesteps): 
            actions = agent.act(states) 

            env_info    = env.step(actions) 
            next_states = env_info[brain_name].vector_observations
            rewards     = env_info[brain_name].rewards
            dones       = env_info[brain_name].local_done
            
            agent.step(states, actions, rewards, next_states, dones, timestep) 

            states = next_states 
            agents_scores += rewards

            if True in dones:
                break
        
        avg_score = np.mean(agents_scores)
        scores_deque.append(avg_score) 
        scores.append(avg_score) 
        
        if ith_episode % 10 == 0:        
            print("Episode: {}\t Average score: {}\t Score: {}".format(ith_episode, np.mean(scores_deque), avg_score)) 
            avg_scores.append(np.mean(scores_deque))
            wandb.log({
                    "episode":ith_episode,
                    "score": avg_score,
                    "moving_average_score": np.mean(scores_deque)
                })

        if ith_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")   
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth") 
            print("Episode: {}\t Average score: {}".format(ith_episode, np.mean(scores_deque))) 

    return scores, avg_scores

scores, avg_scores = ddpg(num_episodes=settings["num_episodes"], max_timesteps=settings["max_timesteps"]) 

fig = plt.figure() 
ax = fig.add_subplot(111) 
#plt.plot(np.arange(1, len(scores)+1), scores) 
plt.plot(np.arange(1, len(avg_scores)+1), avg_scores)
plt.ylabel('Average Score') 
plt.xlabel('Episode #') 
plt.show() 