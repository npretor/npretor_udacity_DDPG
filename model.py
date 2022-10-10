import torch 
import torch.nn as nn
import torch.nn.functional as F

class DDPG_model(nn.Module):
    def __init__(self, input_size, output_size, seed) -> None:
        super(DDPG_model, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

    def forward(self, state): 
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x 