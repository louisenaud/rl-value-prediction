"""
Project:    rl-value-prediction
File:       sarsa.py
Created by: louise
On:         27/04/18
At:         3:50 PM
"""
import gym
import numpy as np


def choose_action(observation, q_table):
    return np.argmax(q_table[observation])


def main():

    env = gym.make("FrozenLake-v0")

    alpha = 0.4
    gamma = 0.999

    q_table = dict([(x, [1, 1, 1, 1]) for x in range(16)])
    score = []

    for i in range(10000):
        observation = env.reset()
        action = choose_action(observation)

        previous_observation = None
        previous_action = None

        t = 0

        for t in range(2500):
            observation, reward, done, info = env.step(action)

            action = choose_action(observation)

            if previous_observation is not None:
                q_old = q_table[previous_observation][previous_action]
                q_new = q_old
                if done:
                    q_new += alpha * (reward - q_old)
                else:
                    q_new += alpha * (reward + gamma * q_table[observation][action] - q_old)

                new_table = q_table[previous_observation]
                new_table[previous_action] = q_new
                q_table[previous_observation] = new_table

            previous_observation = observation
            previous_action = action

            if done:
                if len(score) < 100:
                    score.append(reward)
                else:
                    score[i % 100] = reward

                print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(i, t, reward, np.mean(score)))
                break


if __name__=="__main__":
    main()
