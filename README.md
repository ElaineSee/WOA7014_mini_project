# Introduction
Yee Ling's Mini project for WOA7014: Robotics &amp; Intelligent Systems.

This project is a simple DQN solution for the OpenAI Gym Mountain Car-v0 Environment (https://github.com/openai/gym/wiki/MountainCar-v0).

To solve this environment, we need to find a way to get the car to the goal with as less steps as possible. 

* This environment has a 2D continuous state space (car position & velocity) and 1D Discrete Action Space (to push the car left, right or don't push). 
* For each time step taken, the reward -1. 
* Each episode starts with the car at a random position between [-0.6, -0.4] with a zero velocity 
* Each episode ends either when the car reaches position = 0.5 (the goal) or 200 iterations has been done.

### Solution
This project uses Deep Q-Learning/Deep Q-Network to solve the (discrete) Mountain Car environment. Deep Q-Learning is a model-free, value-based reinforcement learning method. In this method, we use the epsilon-greedy method to increase the exploitation exponentially over time, but the epsilon is floored at 0.1. There's two hidden layers in the neutral network employed, with ReLU activation. 


### Installation
```bash
Gym 0.18.0
Tensorflow  2.3.0
```

### Output
Below are some of the outputs obtained by switching the parameters.
* Original: 32 nodes, 600 episodes, batch size = 64


* Alter to 64 nodes

* Alter to batch size = 10

* Alter to 2000 episodes

### Credit
This program is inspired by John King (https://jfking50.github.io/mountaincar/)
