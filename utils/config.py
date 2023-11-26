import torch

class Config:
    def __init__(self):
        self.action_size = 2
        self.state_size = 24
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.BUFFER_SIZE = int(1e6)  # replay buffer size
        self.BATCH_SIZE = 256 # minibatch size
        self.GAMMA = 0.99           # discount factor
        self.TAU =  1e-3            # for soft update of target parameters
        self.LR = 1e-3            # learning rate 
        self.seed = 100
        self.num_agents = 2