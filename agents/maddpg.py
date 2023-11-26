import numpy as np
import torch

from utils import soft_update
from .ddpg import DDPGAgent

class MADDPG:
    def __init__(self,  config):
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
        """get actions from all agents in the MADDPG object"""
        actions = np.zeros((self.num_agents, self.action_size))
        for i in range(self.config.num_agents):
            agent = self.maddpg_agent[i]
            obs = obs_all_agents[:,i*24:(i*24 + 24)]
            actions[i,:] = agent.act(obs, add_noise, noise_decay)
            
        return actions
    
    def reset(self):
        for i in range(self.num_agents):
            self.maddpg_agent[i].reset()
    
    def actions_target(self, obs_full):
        with torch.no_grad():
            actions = torch.empty((self.BATCH_SIZE, self.num_agents, self.action_size),device=self.config.device)
            for idx, agent in enumerate(self.maddpg_agent):
                obs_i = obs_full[:,idx*24:(idx*24 + 24)]
                actions[:,idx] = agent.actor_target(obs_i)
        return actions
    
    def update(self, agent_i, sample,logger):
        """update the critics and actors of all the agents """
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
        critic_loss.backward()
        agent.critic_optimizer.step()

        current_obs_i = obs_all_agents[:,agent_i*24:(agent_i*24 + 24)]
        action_local = agent.actor_local(current_obs_i)
        if agent_i == 0:
            action_local = torch.cat((action_local, actions[:,1,:]), dim=-1)
        else:
            action_local = torch.cat((actions[:,0,:], action_local),dim=-1)
        action_local = action_local.view(self.BATCH_SIZE,-1)
        actor_loss = -agent.critic_local(obs_all_agents, action_local).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars(f'agent-{agent_i}/losses',
                            {'critic loss': cl,
                            'actor_loss': al},
                            self.iter)

        soft_update(agent.actor_local, agent.actor_target, self.config.TAU)

        soft_update(agent.critic_local, agent.critic_target, self.config.TAU)