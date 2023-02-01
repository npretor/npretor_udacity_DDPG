import random, time, json
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
#from ddpg_agent import Agent 
from ddpg_multi_agent import Agent
import wandb

# Hyperparameters
with open("hyperparameters.json", 'r') as f:
    settings = json.load(f)


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

agent = Agent(state_size=state_size, action_size=action_size, random_seed=10, settings=settings, num_agents=num_agents) 

def ddpg(num_episodes, max_timesteps=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    avg_scores = []
    max_score = -np.Inf 
    
    for ith_episode in range(1, num_episodes+1): 
        """
        Reset the environment 
        Get the starting states
        """
        startTime = time.time() 

        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset() 
        agents_scores = np.zeros(num_agents) 
        states = env_info.vector_observations 
        currentTimesteps = 0

        for timestep in range(max_timesteps): 
            #import ipdb; ipdb.set_trace()
            actions = agent.act(states) 
            env_info    = env.step(actions) 
            next_states = env_info[brain_name].vector_observations
            rewards     = env_info[brain_name].rewards
            dones       = env_info[brain_name].local_done
            
            #for n in range(num_agents):
            #    agent.step(states[n], actions[n], rewards[n], next_states[n], dones[n], timestep) 
            agent.step(states, actions, rewards, next_states, dones, timestep)

            states = next_states 
            agents_scores += rewards
            currentTimesteps = timestep
            if True in dones:
                break
        
        avg_score = np.mean(agents_scores) 
        scores_deque.append(avg_score) 
        scores.append(avg_score) 
        agent.current_avg_score = np.mean(scores_deque)    # Maybe as we get closer lower the Gamma proportionally as well? 

        print("Episode ", ith_episode ," duration: ", int(time.time() - startTime), "   Timesteps: ", currentTimesteps)
        
        if ith_episode % 10 == 0:        
            print("Episode: {}\t Average score: {}\t Score: {}".format(ith_episode, np.mean(scores_deque), avg_score)) 
            
            if np.mean(scores_deque) >= 30.0:
                print("success, saving model")
                torch.save(agent.actor_local.state_dict(), "success_checkpoint_actor.pth")   
                torch.save(agent.critic_local.state_dict(), "success_checkpoint_critic.pth") 
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

# fig = plt.figure() 
# ax = fig.add_subplot(111) 
# plt.plot(np.arange(1, len(scores)+1), scores) 
# plt.plot(np.arange(1, len(avg_scores)+1), avg_scores)
# plt.ylabel('Average Score') 
# plt.xlabel('Episode #') 
# plt.show() 