

## Sequence of events 
1. Initialize the environment 

Start training episodes. This looks like the agents choosing an action, having the environment respond. The environment sends back a state, reward, and the agent gets that result. And then the whole cycle repeats. Simple right?  Just three steps. Let's break those steps down 


1. The agent chooses an action 
2. The environment responds 
3. The agent gets the result 

 
## 1. The agent chooses an action 
When we start, the main actor gets an action based on the current state. however, the network is randomly initialized, so not much should happen. We then train the main actor on what just happened, and add a bit of random noise, not sure why yet or what we are training on.  




## 2. The environment responds 
## 3. The agent gets the result 



DDPG uses 4 networks: 

1. The policy network 
2. The target policy network 
3. Deterministic policy function (?approximator?)  
4. Target policy network 

The actor maps states to actions.   State -> ideal_action    
The critic maps actions to values.  State -> expected_reward  
Target networks: these learn slower, they follow along the main network. 
 > The main network without a target network would likely diverge



Actor (Policy) & Critic (Value)

## Hyperparameters

### Buffer size 

### Batch size 

### Gamma 

### Tau 

### Actor learning rate 

### Critic learning rate 

### Weight decay 

### Number of episodes 

### Network shape 

