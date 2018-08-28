from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from functions.dqn_agent import Agent
from collections import deque
import torch

class DQN():
    """DQN implementation"""
    
    def __init__(self, agent, env, brain, n_episodes=2000, max_t=100000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, verbose = False):
        """Initialize parameters and build model.
        Params
        ======
            agent : trainable agent object
            env : unity environment object
            brain : unity brain object
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            verbose (bool): print information during training
        """
        self.agent = agent
        self.env = env
        self.brain = brain
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.verbose = verbose
        
    def print(self, string, end='\n'):
        """Print information if verbose mode is on
        Params
        ======
            string (string): string to print
            end (string): termination
        """
        if self.verbose:
            print(string, end)
            
    
    def train(self, weights_location = ''):
        """Training

        Params
        ======
            weights_location (string): Location and name of the weights
        """
        if not weights_location:
            weights_location = 'checkpoint.pth'
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start               # initialize epsilon
        for i_episode in range(1, self.n_episodes+1):
            env_info = self.env.reset(train_mode=True)[self.brain.brain_name] # reset the environment
            state = env_info.vector_observations[0]                      # get the current state
            score = 0                                                    # initialize the score
            for t in range(self.max_t):
                action = self.agent.act(state, eps)
                env_info = self.env.step(action)[self.brain.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]        # get the next state
                reward = env_info.rewards[0]                        # get the reward
                done = env_info.local_done[0]                       # see if episode has finished
                score += reward                                     # update the score
                self.agent.step(state, action, reward, next_state, done) # store features in the replay buffer
                state = next_state                                  # roll over the state to next time step
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(self.eps_end, self.eps_decay*eps) # decrease epsilon
            self.print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                self.print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        torch.save(self.agent.qnetwork_local.state_dict(), weights_location)
        return scores