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
        scores_avg[idx] = np.mean(scores[idx-N+1:idx+1])
        scores_std[idx] = np.std(np.transpose(scores[idx-N+1:idx+1]))
    return scores_avg, scores_std

def scores_plot(scores, plot_num, threashold, title = ""):
    """Plotting the scores over episodes for different networks
    
    Params
    ======
        scores (list of dict): Containing at least 'scores', 'fc1_units', and 'fc2_units' for plotting
        plot_num (int): dimensionality of plot (is N for an NxN plot)
        threashold (double): threashold value which is draw in
        title (string): The title of the plot
    """
    f, axs = plt.subplots(plot_num, plot_num, figsize=(20,10), sharex=True, sharey=True)
    cnt = 0
    eval_threashold_idx = np.zeros((plot_num,plot_num))
    eval_max_score = np.zeros((plot_num,plot_num))
    for idx in np.arange(0,plot_num):
        for idy in np.arange(0,idx+1):
            ax = axs[idy, plot_num - idx - 1]
            if idy == 0:
                ax.set_title(str(scores[cnt]['fc1_units']))
            if idx == plot_num - 1:
                ax.set_ylabel(str(scores[cnt]['fc2_units']))
            scores_ = scores[cnt]['scores']
            scores_avg, scores_std = scores_stat(scores_, 100)
            eval_max_score[idy, idx] = np.nanmax(scores_avg)
            with np.errstate(invalid='ignore'): # ignore the nan values when comparing
                eval_threashold_idx[idy, idx] = np.nanargmax(scores_avg > threashold)
            ax.plot(np.arange(len(scores_)), scores_, linestyle='-', color='cornflowerblue', linewidth=1)
            ax.plot(np.arange(len(scores_)), scores_avg, linestyle='-', color='red', linewidth=2)
            ax.plot([0, len(scores_)-1], [threashold, threashold], linestyle='--', color='black', linewidth=2)
            cnt = cnt + 1
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0)
    f.suptitle(title)
    return np.fliplr(eval_threashold_idx), np.fliplr(eval_max_score)