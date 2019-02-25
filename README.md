Collaboration and Competition Project
=====================================

This repository is a submission for the collaboration and competition project of Udacity's deep reinforcement learning course.


Project Details
---------------

The assignment is to solve the Unity tennis environment.  The environment is considered solved when the average maximum score for both agents reaches 0.5 per episode.  I solved the environment with an implementation of PPO with separate actor and critic networks.  My PPO implementation was modified to accommodate multiple agents in a collaborative or competitive environment using the same technique used by Lowe et al. in MADDPG.  Specifically, each agent has a separate policy network and the training process uses a shared value network as a critic during training.  The shared value network receives all observations and actions from both agents, thus observing a stationary environment.  This value function then stabilizes training for the policy networks, each of which operates in an environment containing another learning agent.  Both policy networks had identical configurations, with a single hidden layer of 400 nodes, relu activation functions, and a normally distributed policy parameterized by its mean and standard deviation.  The value network also used relu activation functions and a single hidden layer of 400 nodes.  


Getting Started
---------------

This repository is designed to be easy to set up on all supported platforms, but was only tested on OSX nad Linux.  Make sure you are using an anaconda python distribution, clone the repository, then create the project environment by running:

    conda env create

from the repositories root directory.  This will create the drlnd_p3 environment, which can then be
activated by:

    source activate drlnd_p3
    
The drlnd_p3 environment contains the following packages:

    - pytorch
    - python=3.6.5
    - nose=1.3.7
    - numpy=1.14.5
    - matplotlib=3.0.2
    - pytorch=0.4.0
    - protobuf=3.5.2
    - toolz=0.9.0
    - multipledispatch=0.6.0
    - infinity=1.4
    - six=1.11.0
    - tqdm=4.28.1
    - pandas=0.23.4
    - scipy=1.1.0
    - unityagents=0.4.0

Next, you can run the project's limited unit tests by running:

    nosetests

The project should download and extract the reacher environment automatically.  If it fails, please place the downloaded and extracted environment file in mappo/environment/resources and try the tests again.  It may be necessary to fix the permissions of the extracted file.


Instructions
------------

Once the above environment has been installed and activated and the tests pass, a new agent can be created and trained by running train.py:

    python train.py

The train.py script will train a new agent and save that agent as checkpoint.pth.  A plot of rewards will be displayed after training, and the plot will be saved to plot.png.
 
    
