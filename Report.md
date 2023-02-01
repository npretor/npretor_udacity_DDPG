

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
The agent get the resulting action and reward and adds it to a list of memories. The <b>batch size</b> is the number of actions to wait before training. 


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
The learning rate is the amount by which the network nodes are updated. A common default value would be 0.1, meaning the weights of the network are updated by one tenth of the estimated weight error. 
A larger learning rate allows for faster learning, but means a solution will be arrived at which is probably less than optimal. A smaller learning rate will learn slowly, and too small and the model will oscillate over training periods

### Critic learning rate 

### Weight decay 

### Number of episodes 

### Network shape 


## Training experience 
Early training episodes I trained for 20. No success, no score above 4.
Tried training with single agent training, one episode reached 14, then fell down to 8 and kept falling after 5000 episodes. 
Then wrote a multi-agent training script and agent. 
(fluent-oath-2) Tried on training session for 1500, peaked at 5, then oscillated up and down between 3 and 4 with a period of about 500 episodes peak to peak. 
(noble-blaze-3) Segfault 
(fluent-dust-4) Reduced the critic and actor learning rates from 10^-3 to 10^-4, hoping this prevents oscillations. Segfault at 700 episodes
(rural-spaceships-5) Restarted training, changed gamma from 0.95 to 0.99. Peaked at 2.5 early, then gradually declined to 1. 
(charmed-wave-6) Changed to learn every 4 episodes instead of every episode. Slow linear growth up to .44. 

Found out I was not connecting the state and action into the Critic. 

(wandering-dust-32) Fixed an issue where i was training each timestep. After 250 episodes the score dropped down to 1.8 
(graceful-morning-33) Re-enabled gradient clipping. No score difference at all on the score up to 130 episodes 
(dutiful-thunder-35)  Found an issue where the noise for one agent was being added to all, resulting in non-uniform noise sampling.  

(denim-lake-37) Disabled weight decay, previously was set to zero 

(lyric-armadillo-41) Weight decay was set to 999 accidentally... 
(42) Disabled custom weight decay. Never got above 0.5
(43) Nothing LR-A 1e-4    LR-C 1e-3 
(44) Nothing LR-A 1e-3    LR-C 1e-4 
(45) LR-A 1e-3    LR-C 1e-3 
(46) Batch size 256
(47) Batch size 1024 - max is 25 
(48) Made tau larger by 10x. No learning, scores at 0.003 
(50) Changed network shape to 256-128 and 256-128 
(51) network shape is 128-64, 128-64

(54) Udacity default values. No gradient clipping 
(55) Gradient clipping for both nets 
(57, 57) Same parameters, different seed value 
(63) Trained to 4000, close to 30 but never reached
(64) Failed with SIGBART at 3000 episodes  
(65) Code bug 
(66) Added noise scaling which should decrease noise(explore factor) as proximity to the goal approaches 
