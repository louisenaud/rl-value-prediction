"""
Project:    rl-value-prediction
File:       td0.py
Created by: louise
On:         27/04/18
At:         2:17 PM
"""
import gym
import numpy as np


def main():
    # Create and render Frozen Lake OpenAI gym environment
    env = gym.make('FrozenLake-v0')
    env.render()
    # generate a random distribution of the observations
    p = np.random.dirichlet(np.ones(4), size=16)
    print(p.shape)

    nb_episodes = 5000
    alpha = 0.01
    gamma = 0.9
    v = np.zeros(16)
    for i in range(nb_episodes):
        done = False
        observation = env.reset()  # first observation, agent at S
        while not done:  # while agent is not at G
            # sample an action from the probability distribution of the observation (fixed policy)
            action = np.random.choice(4, 1, p=p[observation])
            # Take action
            new_observation, reward, done, info = env.step(action[0])
            # TD(0) update
            v[observation] = v[observation] + alpha * (reward + gamma * v[new_observation] - v[observation])
            # update observation
            observation = new_observation
    print(v)


if __name__=="__main__":
    main()
