import numpy as np
import torch
from utils import soft_update
from .ddpg import DDPGAgent
import wandb

class MADDPG:
    """
    Multi-agent deep deterministic policy gradient (MADDPG) class that trains a group of agents to perform a cooperative task.

    Args:
        config: A configuration object containing hyperparameters for training the MADDPG agents.

    Attributes:
        maddpg_agent (list): A list of DDPGAgent objects representing each individual agent in the MADDPG group.
        num_agents (int): The number of agents in the group.
        action_size (int): The size of the action space for each agent.
        state_size (int): The size of the state space for each agent.
        seed (int): The random seed used for training.
        BATCH_SIZE (int): The batch size used for training.
        GAMMA (float): The discount factor used for computing returns.
        iter (int): The current iteration number of the training process.
    """

    def __init__(self, config):
        self.config = config
        self.maddpg_agent = [DDPGAgent(config), DDPGAgent(config)]
        self.num_agents = len(self.maddpg_agent)
        self.action_size = config.action_size
        self.state_size = config.state_size
        self.seed = config.seed
        self.BATCH_SIZE = self.config.BATCH_SIZE
        self.GAMMA = self.config.GAMMA
        self.iter = 0

    def act(self, obs_all_agents, add_noise=False, noise_decay=1.):
        """
        Gets the actions of all agents in the MADDPG object.

        Args:
            obs_all_agents (numpy.ndarray): A 2D numpy array containing the observations of all agents in the group.
            add_noise (bool, optional): A boolean flag indicating whether or not noise should be added to the action outputs.
            noise_decay (float, optional): A scalar value used to decay the amount of noise added over time.

        Returns:
            numpy.ndarray: A 2D numpy array containing the actions of all agents in the group.
        """
        actions = np.zeros((self.num_agents, self.action_size))
        for i in range(self.config.num_agents):
            agent = self.maddpg_agent[i]
            obs = obs_all_agents[:,i*self.state_size:(i*self.state_size + self.state_size)]
            actions[i,:] = agent.act(obs, add_noise, noise_decay)
            
        return actions
    
    def reset(self):
        """
        Resets the noise process for each agent in the MADDPG object.
        """
        for i in range(self.num_agents):
            self.maddpg_agent[i].reset()
    
    def actions_target(self, obs_full):
        """
        Gets the target actions of all agents in the MADDPG object.

        Args:
            obs_full (torch.Tensor): A 2D tensor containing the observations of all agents in the group.

        Returns:
            torch.Tensor: A 3D tensor containing the target actions of all agents in the group.
        """
        with torch.no_grad():
            actions = torch.empty((self.BATCH_SIZE, self.num_agents, self.action_size),device=self.config.device)
            for idx, agent in enumerate(self.maddpg_agent):
                obs_i = obs_full[:,idx*24:(idx*24 + 24)]
                actions[:,idx] = agent.actor_target(obs_i)
        return actions
    
    def update(self, agent_i, sample, logger=None):
        """
        Updates the critics and actors of all agents in the MADDPG object.

        Args:
            agent_i (int): The index of the agent to be updated.
            sample (tuple): A tuple containing the observations, actions, rewards, next observations, and dones for all agents in the group.
            logger (wandb.Run, optional): A wandb run object for logging training metrics (default=None).
        """
        obs_all_agents, actions, rewards, next_obs_all_agents, dones = sample

        agent = self.maddpg_agent[agent_i]

        next_actions = self.actions_target(next_obs_all_agents)
        next_actions = next_actions.view(self.BATCH_SIZE,-1)
        
        with torch.no_grad():
            q_next = agent.critic_target(next_obs_all_agents, next_actions)
        y = rewards[:,agent_i].view(-1, 1) + self.GAMMA* q_next * (1 - dones[:,agent_i])

        actions_full = actions.view(self.BATCH_SIZE, -1)
        q_preds = agent.critic_local(obs_all_agents, actions_full)

        critic_loss = (q_preds - y.detach()).pow(2).mul(0.5).sum(-1).mean()
        agent.critic_optimizer.zero_grad()
        critic_loss
