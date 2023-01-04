


# Overview of what's happening 


1. Initialize the environment 
2. Initialize critic network and actor network 
3. Initialize target network and 

Several types of networks: 
State -> action_probability mapping. In this case the actions can be selected 
State -> action mapping 

Suppose we have an agent with a goal. It does a series of things, and at the end of an episode gets a reward. The problem is we can miss a lot of things even in episodes that don't give rewards, we might have good behaviors. We need a way to determine if we are closer or farther from our goal


Policy gradient methods are very similar to supervised learning 
We train on sequences of (state, action) pairs, with loss as the reward 
In a traditional neural network, we have a input, and a label. The neural network starts randomized, but over time


Policy network: This network takes that state of the game and outputs an action probability


Trajectory: state-action sequence. The reason we use these and not just episodes is we can use trajectories for both episodic and continuing tasks
Tao: sequence of states and actions 
R(t) is total expected return of this action sequence 


Use the policy to collect trajectories 
It's like detecting the summit of a mountain by searching locations on a mountain when blind, but having the ability to teleport. You sample north, south, east, and west, and determine which of those samples are the steepest(most positive). Move in that direction. 

## My gradient ascent of Everest

Imagine if you will, you are a time traveller, cast back into the misty ancient past of 2010. You were blinded by the explosion of the caused by an instability in the bose-einstein condensate of (130-proton, 130-electron, 132-neutron) that powers your travel pack, and you need to get to a high point on earth. You can randomly teleport to locations, but you have to feel your way around to find the peak of Mount Everest, where Sir Edmund Hillary left a bunch of radium to keep warm when he summited Everest. You can use this to jump-start your time travel device, which has been downgraded to just a spatial teleporter. 

You start off. You are blind, and in order to get a sense of the lay of the land, you start random jumps, each time noting the (lat, long) tuple of where you land, as well as the elevation. When you teleport, you sense the north, south, east, and west, and since it's a sunny day on everest, you don't get disoriented by sensing the sun's heat. After sensing the height and the direction of the slope of the mountain, you start to create a 3D haptic model of the mountain on your iPad, a gold plated 1000th anniversary edition. At first you start off with 1000m jumps randomly in any direction. After a time though, you have jumped around, and started to get a feel for the larger landscape of the mountain, so you adjust your teleporter to jump in smaller distances. As you continue teleporting and decreasing the distance gradually, you begin to build up a rough map of the mountain, and can start directing your teleporter towards large features, searching those with small jumps. Finally you start samping in the meters, and after a few jumps you arrive on the mountain's peak. All of your north, south, east, and west groping in the dark steeply fall away into the dark, and you know you are at the summmit. You feel the heat from the radium, which is barely into it's first half-life, rip open it's protective canister, and put it into your matter processor to be converted. Success! While you rest you take a moment to chat with the stunned summiters that have just arrived, and are wondering why you are wearing a bodysuit that absorbs all light normal to it's surface. As you tell them your story, you notice one of the party is only half-listening, or maybe listening but transfixed by some inner thoughts. As you leave you shake his hand, and ask him his name. "Andrej Karpathy and this is my companion, Demis Hassabis, pleased to meet you". As space and time fold around you, you wonder if this will merit a disciplinary hearing, since time travellers are not permitted to show their technology. You think to yourself, "Nah, they couldn't have learned anything of value from me, the tech is too advanced." Besides, the disclosure regulations only apply to civilizations closer than 100 years apart temporally, otherwise the technological gaps between them create such a chasm that meaningfully talking about scientific discoveries sounds like gibberish anyways.

Five years later on Earth, Deepmind is built 




### So how does this method work? 
Greedy strategy = Exploit 
Random? strategy = Explore only

Say you are balancing on a onewheel. The state is represented by a few things: 
- angle of the board (theta) 
- velocity (x, y, z) 
- acceleration (x, y, z)  

Or, you want to create a plane stabilizer 
Start with safe mode: You have a fixed range of attitutes the plane can do: 
No stall angles 
No dive angles lower than 45 degrees 
No rolls over 45 degrees 
Inputs are the stick, outputs are the control surfaces 


### Things I learned: 

### Tuning the learning rate. 
Too fast of a learning rate and the model will converge then diverge 
https://stats.stackexchange.com/questions/324896/training-loss-increases-with-time


* Choosing a network size
* Batch size size matters a lot, a larger batch size greatly helped convergence


### Using clipping when there is no convergence
https://machinelearningmastery.com/exploding-gradients-in-neural-networks/


### CHoosing a learning rate 
https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0 
