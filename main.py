# main function that sets up environments
# perform training loop

import os
import random
from collections import deque

import torch
import numpy as np
import wandb  # Import wandb

from unityagents import UnityEnvironment
from utils import Config, ReplayBuffer
from agents import MADDPG 

env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print("brain name:", brain_name)
print("brain:", brain)

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

LEARN_NUM = 5 # number of learning passes

def main():
    n_episodes=2000
    max_t=10000

    configs = Config()
    maddpg = MADDPG(configs)
    memory = ReplayBuffer(configs)
    scores_window = deque(maxlen=100)
    scores_all = []
    moving_average = []
    best_score = -np.inf
    PRINT_EVERY = 10
    SOLVED_SCORE = 0.5

    # Initialize wandb run
    wandb.init(project="tennis-maddpg", entity="minhna1112")  # Replace 'your_username' with your wandb username

    model_dir= os.getcwd()+"/model_dir"
    os.makedirs(model_dir, exist_ok=True)
    START_NOISE_DECAY = 5.
    noise_decay = START_NOISE_DECAY

    for i_episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]         
        maddpg.reset()
        states = env_info.vector_observations          
        states = np.reshape(states,(1,num_agents*state_size))
        scores = np.zeros(num_agents) 

        for episode_t in range(max_t):
            actions = maddpg.act(states, add_noise=True, noise_decay=noise_decay)
            env_info = env.step(actions)[brain_name]         
            next_states = env_info.vector_observations        
            next_states = np.reshape(next_states,(1, num_agents*state_size))
            rewards = env_info.rewards                       
            dones = env_info.local_done                      
            actions = np.expand_dims(actions,axis=0)
            memory.add(states, actions, rewards, next_states, dones)
            if len(memory) > configs.BATCH_SIZE:
                for a_i in range(num_agents):
                    samples = memory.sample()
                    maddpg.update(a_i, samples, None)  # Removed logger from update function
                maddpg.iter += 1

            scores += rewards                        
            states = next_states                             
            if np.any(dones):                                 
                break
        
        ep_best_score = np.max(scores)
        scores_window.append(ep_best_score)
        scores_all.append(ep_best_score)
        moving_average.append(np.mean(scores_window))

        # save best score                        
        if ep_best_score > best_score:
            best_score = ep_best_score
        
        # Logging with wandb
        wandb.log({'Episode': i_episode, 'Max Reward': np.max(scores), 'Moving Average': moving_average[-1]})

        if i_episode % PRINT_EVERY == 0:
            print('Episodes {:0>4d}-{:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}\tNoise decay: {:.3f}'.format(
                i_episode-PRINT_EVERY, i_episode, np.max(scores_all[-PRINT_EVERY:]), moving_average[-1], noise_decay))

        # determine if environment is solved and keep best performing models
        if moving_average[-1] >= SOLVED_SCORE:
            #saving model
            print('Episode {:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
                i_episode, ep_best_score, moving_average[-1]))
            save_dict_list =[]
            for i in range(num_agents):
                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor_local.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic_local.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)
                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(i_episode)))
            break

        if noise_decay > 0.020:
            noise_decay = START_NOISE_DECAY / (1. + i_episode)

    wandb.finish()  # Close the wandb run
    env.close()

if __name__=='__main__':
    main()
