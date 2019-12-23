# DRLND_Tennis
Solution to project 3 (Collaboration and Competition) in the Deep Reinforcement Learning Udacity Nanodegree
This solution solves the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment from the Unity ML-Agents Toolkit.

This repository contains:

- `Report.pdf`: provides a description of the solution including the following:
	- A description of the learning algorithm.
	- A description of the neural networks used.
	- A plot of the rewards per episode to illustrate that the agent is improving performance over time.
	- Ideas for future work.
- `run.ipynb`: A jupyter notebook that either trains the agent in the Unity environment or loads a set of pre-trained weights to evaluate the agent's performance.
- `ddpg`: A directory containing the implementations of DDPG and MADDPG used to solve the environment. The contents of the directory are:
	- `results/`: A directory containing:
		- .txt files with the scores of training agents with different parameters.
		- PyTorch serialization of learned model parameters from a training sessions of well-performing agents. These can be loaded into the agent to evaluate agent performance without needing to train the agent.
	- `src/`: A directory containing the various python files imported in the python notebook used for training, defining models, etc.
		- `agent.py`: This python file contains the defines for the hyperparameters used for the agent.
- `images/`: A directory containing:
	- Plot of the scores of agents training

## Contents

1. [Project Details](#project-details)
2. [Getting Started](#getting-started)
3. [Instructions](#instructions)


## Project Details
In this environment, two tennis players with a racker each must bounce the ball over the net and avoid hitting the ground. 
A reward of +0.1 is provided for an agent if the agent manages to bounce the ball over the net. A reward of -0.01 is provided for an agent if the ball hits the ground
Thus, the goal of each of the agents is to bounce the ball back and forth for as many times as possible.
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. However the information is stacked three times inside the unity environment so the final state representation consist of a total of 24 states. 
Each action is a vector with two numbers, corresponding to racket horizontal and vertical movement. 
Every entry in the action vector should be a number between -1 and 1.
I this environment there are 2 agents, the two tennis players. 

The agents must get an average score of +0.5 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, the rewards that each agent received (without discounting) are added up, to get a score for each agent.  This yields 2 (potentially different) scores.  We then take the average of these 2 scores. 
- This yields an **average score** for each episode (where the average is over all 2 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least 0.5. 

## Getting Started

1. Follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to setup a python environment. 
By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

2. Download the unity environment from one of the links below.  You need only select the environment that matches your operating system:

	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Unzip (or decompress) the file.

4. Clone this repository and change the path for the unity environment in the `run.ipynb` notebook to whereever you placed the unity environment. 
 

## Instructions

Run the code cells in `run.ipynb` to get started with training the agent. 
The time taken to train the agent depends on the specific computer and whether it has a CUDA-graphics processor available or not. 
Additionally, in the bottom of the notebook it is showed how pretrained weights can be loaded into the agent to evaluate the agents performance.