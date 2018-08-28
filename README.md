# udacity_project_navigation

## Project Details

This document is the report of the first Udacity project for reinforcment learning. 

For this project, I have trained an agent to navigate and collect yellow bananas in a large, square world.

![example banana](images/banana.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

### Step 1: Clone the DRLND Repository

If you haven't already, please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.
These instructions can be found in `README.md` at the root of the repository.
By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10.
While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions.
Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Step 2: Download the Unity Environment

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below.
You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
Then, place the files from this repository in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment.
You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

### Step 3: Explore the Environment

After you have followed the instructions above, open `Report.ipynb` (located in the `p1_navigation/` folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.

## Instructions

The whole logic of this project lives in `Report.ipynb`.
Therefore, this is the only file which needs to be reviewed and exectued.
Afterwards, the folder structure of this project is listed for further details:

	.
	├── eval                                    # contains pickeld evaluated scores for the corresponding networks
	│   ├── dict_fc1-16_fc2-16_e-400_decay-0.995
	│   ├── ...
	│   └── dict_fc1-8_fc2-8_e-2000_decay-0.995
	├── functions                               # DQN and learning functions
	│   ├── dqn_agent.py                        # Implements the DQN algorithm
	│   ├── helper.py                           # Some neat helper functions
	│   ├── __init__.py
	│   ├── model.py                            # Allocating the Q-Network with two hidden layers
	│   └── rl.py                               # Implements the training of the Q-Network 
	├── images                                  # Image folder
	├── LICENSE
	├── python                                  # Amended version of the python folder from the ML-Agents repository
	├── README.md                               # Project readme file
	├── Report.ipynb                            # The main project report
	└── weights									# contains the weights of each trained network with various hyper parameters
	    ├── network_fc1-16_fc2-16_e-400_decay-0.995.pth
	    ├── ...
	    └── network_fc1-8_fc2-8_e-2000_decay-0.995.pth


