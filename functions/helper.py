import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def env_info(env, brain):
    """Prints and returns environment information
    
    Params
    ======
        env : unity environment object
        brain : unity brain object
    """

    # reset the environment
    env_info = env.reset(train_mode=True)[brain.brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)
    return state_size, action_size


def scores_stat(scores, N):
    """Calculates average and standard deviation over the last N epsiodes of the scores
    
    Params
    ======
        scores (np.array): vector of scores per episode
        N (int): window size
    """

    scores_avg = np.nan * np.zeros((len(scores),))
    scores_std = np.nan * np.zeros((len(scores),))
    for idx in np.arange(N-1, len(scores)):
        scores_avg[idx] = np.mean(scores[idx-N-1:idx+1])
        scores_std[idx] = np.std(scores[idx-N-1:idx+1])
    return scores_avg, scores_std