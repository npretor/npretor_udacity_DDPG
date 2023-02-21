# npretor_udacity_DDPG
Deep Deterministic Policy Gradient implimentation for Udacity DeepRL course

## Setup 
1. Install conda 
2. Setup environment 
```
conda create --name drlnd python=3.6 
conda activate drlnd 
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
```
3. Fix an issue with Pytorch
<b>Change the pytorch version in the requirements file to be the latest version, or remove version number altogether </b>
4. Install requirements 
```
pip install .
```

5. Create an IPython kernel for the drlnd environment.
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
6. Start notebook 
```
python3 -m jupyter notebook 
```
7. Select the drlnd kernel 


## Training 
* The working implimentation is the multi-agent training. Activate the environment, disable wandb(unless you have logged in), and run: 
```
conda activate drlnd 
wandb disabled 
cd $YOUR_REPO_INSTALL_LOCATION
python3 multi_agent_training.py
```

### Project overview and environment 
The environment is considered solved 