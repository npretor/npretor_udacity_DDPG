# Taken from: 
# https://www.youtube.com/watch?v=oPGVsoBonLM

import torch

x = torch.tensor(9.0, requires_grad=True)
print(x)

y = x**2
print(y) 

y.backward()
print(x.grad) 


"""
Lets say this equation describes our loss landscape: 

y=x**3-2x+1
 
""" 

