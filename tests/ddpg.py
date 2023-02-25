import pytest
import sys 
sys.path.append('../') 
import json 
from ddpg_multi_agent import Agent   


# = = = = = = = = = = Fixturing = = = = = = = = = = #

@pytest.fixture
def settings():
    with open("../hyperparameters.json", 'r') as f:
        settings = json.load(f)    
    return settings

@pytest.fixture
def ddpg(settings):
    ddpg = Agent(num_agents=2, state_size=33, action_size=2, seed=0, settings=settings)
    return ddpg 
    


# = = = = = = = = = = Function unit tests = = = = = = = = = = #
def test_act(ddpg):
    ddpg.act('agent_states')

