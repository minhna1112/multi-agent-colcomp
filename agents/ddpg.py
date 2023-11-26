import torch
import torch.optim as optim
import numpy as np

from networks import ActorNN, CriticNN
from utils import OUNoise

class DDPGAgent:
    def __init__(self, config):
        """
        Initializes a Deep Deterministic Policy Gradient (DDPG) agent.

        Args:
        - config: a configuration object containing hyperparameters and other settings

        Returns:
        - None
        """
        state_size = config.state_size
        action_size = config.action_size
        seed = config.seed
        num_agents = config.num_agents

        self.device = config.device
        self.BATCH_SIZE = config.BATCH_SIZE
        self.TAU = config.TAU
        self.LR = config.LR
        self.BUFFER_SIZE = config.BUFFER_SIZE
        self.GAMMA = config.GAMMA

        # Actor-Network
        self.actor_local = ActorNN(state_size, action_size, seed).to(self.device)
        self.actor_target = ActorNN(state_size, action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR)
        print("===================== Actor Network =========================")

        #Critic-Network
        self.critic_local = CriticNN(num_agents*state_size, num_agents*action_size , seed).to(self.device)
        self.critic_target = CriticNN(num_agents*state_size, num_agents*action_size , seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR)
        print("===================== Critic Network =========================")

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

         # Noise process
        self.noise = OUNoise(action_size, seed)

        self.t_step = 0

    def hard_copy_weights(self, target, source):
        """
        Copy weights from the source network to the target network.

        Args:
        - target: the target network
        - source: the source network

        Returns:
        - None
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self, state, add_noise=False, noise_decay=1.):
        """
        Returns actions for given state as per current policy.

        Args:
        - state: the current state of the agent
        - add_noise (optional): whether to add noise to the actions
        - noise_decay (optional): the amount by which to decay the noise

        Returns:
        - action: a numpy array with the actions for the given state
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += noise_decay * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """
        Resets the noise process.

        Args:
        - None

        Returns:
        - None
        """
        self.noise.reset()
