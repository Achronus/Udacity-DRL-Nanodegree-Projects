# Project: Navigating a World of Bananas

## Introduction

This project focuses on my solution to creating an agent that automatically navigates a large square world, which is full of bananas. It uses a simulated environment based on [Unity's ML-Agents](https://github.com/Unity-Technologies/ml-agents) and can be found within [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

![Banana World Environment](https://github.com/Achronus/Udacity-DRL-Nanodegree-Projects/blob/master/imgs/bananas.gif)

## Environment Details

The project focuses on a single agent, where it's goal is to collect as many yellow bananas as possible while avoiding the blue bananas. It recieves a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent learns how to best select the appropriate action. Four discrete actions are available, these are:

- 0 - moves forward
- 1 - moves backward
- 2 - turns left
- 3 - turns right

This project uses an episodic approach with a basic Deep Q-Network to solve the environment, where the agent achieves an average score of +13 over 100 consecutive episodes.

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
    cd p1_navigation/python
    conda install pytorch=0.4.0 -c pytorch
    pip install .
    cd ..
    ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

4. Download the environment from one of the links below, by selecting the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

5. Unzip the environment archive into the _'project's environment'_ directory and then adjust the path to the `UnityEnvironment` within the Jupyter Notebook mentioned in the 'How to Use' section.

    _(For full details)_ refer to the Udacity courses GitHub project [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation#getting-started).

## How to Use

The code to train and test the agent within the banana enviornment can be found within the `Navigation.pynb` Jupyter Notebook, where the code cells can be run in sequence. Both sections are labelled accordingly.

To open the notebook use the command `jupyter-notebook`, within the _'project's environment'_ directory. Before running the code in the notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.
