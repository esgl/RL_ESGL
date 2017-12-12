from __future__ import print_function
from __future__ import division

import gym
import numpy as np
# import PIL.Image as Image
import matplotlib.pyplot as plt

from PolicyGradient import PolicyGradient


GAME = 'Pong-v0'
# RENDER = True
RENDER = False
DISPLAY_REWARD_THRESHOLD = -1

env = gym.make(GAME)
print(env.action_space)
print(env.action_space.n)
print(env.observation_space)
print(env.observation_space.shape[0])

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()

if __name__ == '__main__':

    episode = 1000000
    n_action = env.action_space.n
    n_action = 2
    observation_tmp = env.reset()
    observation_tmp = prepro(observation_tmp)
    n_observation_space = observation_tmp.size

    RL = PolicyGradient(n_action, n_observation_space)
    ep_rs_sum_all = []
    for i_episode in range(episode):
        cur_observation = env.reset()
        # cur_observation = cur_observation.ravel()
        cur_observation = prepro(cur_observation)
        pre_observation = None
        score = 0
        while True:
            if RENDER:
                env.render()
            # observation = cur_observation - pre_observation if pre_observation is not None else np.zeros_like(cur_observation)
            observation = cur_observation

            action = RL.choose_action(observation)
            observation_, reward, done, _ = env.step(action + 2)
            # print('reward: ', reward)

            if reward != 0:
                score += 1
                text = '' if reward == -1 else '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                print('eposide %d  score %d, %s' % (i_episode, score, text))

            observation_ = prepro(observation_)

            RL.store_transition(observation=observation_, action=action, reward=reward)
            if done:
                ep_rs_sum = sum(RL.ep_rewards)
                ep_rs_sum_all.append(ep_rs_sum)
                print("\n***********************************************")
                print("eposide %d, reward %d" % (i_episode, ep_rs_sum))
                print("***********************************************\n")
                vt = RL.learning()
                # if i_episode == 30:
                #     plt.plot(vt)
                #     plt.xlabel("episode steps")
                #     plt.ylabel("normalized state-action value")
                #     plt.show()
                break

            pre_observation = cur_observation
            cur_observation = observation_

    # plot
    plt.plot(ep_rs_sum_all)
    plt.xlabel("episode steps")
    plt.ylabel("sum of reward in episode")
    plt.show()
