from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import gym
import numpy as np

import matplotlib.pyplot as plt

from policygradient.PolicyGradientConvNN import PolicyGradientConvNN

GAME = "Pong-v0"
# RENDER = False
RENDER = True

DISPLAY_REWARD_THRESHOLD = -1

env = gym.make(GAME)
print(env.action_space)
print(env.action_space.n)
print(env.observation_space)
print(env.observation_space.shape[0])

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I==144] = 0
    I[I==109] = 0
    I[I!=0] = 1
    return I.astype(np.float32).ravel()

if __name__ == '__main__':
    episode = 1000000
    n_actions = env.action_space.n
    n_actions = 2
    observation = env.reset()
    observation = prepro(observation)
    print(len(observation))

    RL = PolicyGradientConvNN(n_actions=n_actions, n_features=len(observation))

    for i_episode in range(episode):
        observation = env.reset()
        cur_x = prepro(observation)
        pre_x = None
        ep_rs_sum_all = []
        while True:
            if RENDER:
                env.render()

            # observation = cur_x
            action = RL.choose_action(observation=cur_x)
            observation_, reward, done, _ = env.step(action + 2)

            if reward != 0:
                text = '' if reward == -1 else '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                print('eposide %d  reward %d, %s' % (i_episode, reward, text))

            cur_x = prepro(observation_)

            RL.store_transition(s=cur_x, a=action, r=reward)

            if done:
                ep_rs_sum = sum(RL.eps_re)
                ep_rs_sum_all.append(ep_rs_sum)
                print("\n***********************************************")
                print("eposide %d, return %d" % (i_episode, ep_rs_sum))
                print("***********************************************\n")
                vt = RL.learning()
                break
            pre_x = cur_x
            # cur_x = observation_

     # plot
    plt.plot(ep_rs_sum_all)
    plt.xlabel("episode steps")
    plt.ylabel("sum of reward in episode")
    plt.show()