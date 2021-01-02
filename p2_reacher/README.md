# Project: Sticking to the Target

## Introduction

This project focuses on my solution to creating multiple agents that automatically move a double-jointed arm and keep it within a moving target area. It uses a simulated environment based on [Unity's ML-Agents](https://github.com/Unity-Technologies/ml-agents) and can be found within [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

![Reacher](https://github.com/Achronus/Udacity-DRL-Nanodegree-Projects/blob/master/imgs/reacher.gif)

## Environment Details

The project focuses on 20 identical agents, where each one uses a copy of the environment and has a double-jointed arm that must keep to a moving target location. A reward of +0.1 is provided for each step that the agent's hand is in the goal location, where the goal of the agents is to maintain it's position at the target location for as many timesteps as possible.

The state space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Given this information, the agent learns how best to use an action vector containing four continuous actions that correspond to torque applicable to the two joints within the arm. Every entry in the action vector is a number between -1 and 1.

The approach taken uses a DDPG model with the Ornstein-Uhlenbeck process and Experience Replay, solving the environment in 118 episodes with an average score of +30 over 100 consecutive episodes, across all agents.

### Dependencies

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

4. Download the environment from one of the links below, by selecting the environment that matches your operating system:

   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

5. Unzip the environment archive into the _'project's environment'_ directory and then adjust the path to the `UnityEnvironment` within the Jupyter Notebook mentioned in step 6.

    _(For full details)_ refer to the Udacity courses GitHub project [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started).

## How to Use

The code to train and test the agent within the enviornment can be found within the `Continuous_Control.pynb` Jupyter Notebook, where the code cells can be run in sequence. Both sections are labelled accordingly.

To open the notebook use the command `jupyter-notebook`, within the _'project's environment'_ directory. Before running the code in the notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.
