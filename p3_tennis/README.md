# Project: A Competitive Game of Tennis

## Introduction

This project focuses on my solution to creating a competitive environment of two agents playing table tennis. It uses a simulated enviornment based on [Unity's ML-Agents](https://github.com/Unity-Technologies/ml-agents) and can be found within [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

![Tennis](https://github.com/Achronus/Udacity-DRL-Nanodegree-Projects/blob/master/imgs/tennis.gif)

## Environment Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play for as long as possible.

The state space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task focuses on an episodic approach and uses a [MADDPG](https://arxiv.org/abs/1706.02275) model to solve the environment in 442 episodes with an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## Dependencies

This project requires a Python 3.6 environment with PyTorch 0.4.0, which can be created by following the instructions below.

1. Create (and activate) a new environment with Python 3.6.

   - Linux or Mac

    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```

   - Windows

   ```bash
   conda create --name drlnd python=3.6
   activate drlnd
   ```

2. Clone the repository, navigate to the `python/` folder and install the required dependencies.

    _(Note)_ a requirements.txt file is accessible within this folder which details a list of the required dependencies.

    ```bash
    git clone https://github.com/Achronus/Udacity-DRL-Nanodegree-Projects.git
    cd p2a_cc_reacher/python
    conda install pytorch=0.4.0 -c pytorch
    pip install .
    cd ..
    ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

4. Download the environment from one of the links below, selecting the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

5. Unzip the environment archive into the _'project's environment'_ directory and then adjust the path to the `UnityEnvironment` within the Jupyter Notebook mentioned in the 'How to Use' section.

    _(For full details)_ refer to the Udacity courses GitHub project [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started).

## How to Use

The code to train and test the agent within the enviornment can be found within the `Tennis.pynb` Jupyter Notebook, where the code cells can be run in sequence. Both sections are labelled accordingly.

To open the notebook use the command `jupyter-notebook`, within the _'project's environment'_ directory. Before running the code in the notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.
