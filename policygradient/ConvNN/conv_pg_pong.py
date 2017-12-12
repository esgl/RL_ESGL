from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import gym
import numpy as np
import tensorflow as tf
import pickle as pkl

import matplotlib.pyplot as plt

from policygradient.ConvNN.PolicyGradient import PolicyGradient
GAME = "Pong-v0"
RENDER = False
# RENDER = True

DISPLAY_REWARD_THRESHOLD = -1
CHECK_POINT_STEP = 1000
checkpoint_dir = "logs/saver/"
pkl_file = "logs/saver/ep_rs_sum_all.pkl"



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
    observation = env.reset()
    observation = prepro(observation)
    print(len(observation))

    RL = PolicyGradient(n_features=len(observation))

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(RL.sess, ckpt.model_checkpoint_path)
        print("Model restore in file: %s" % ckpt.model_checkpoint_path)

    for i_episode in range(episode):
        observation = env.reset()
        cur_x = prepro(observation)
        pre_x = None
        ep_rs_sum_all = []
        while True:
            if RENDER:
                env.render()
            if pre_x is not None:
                obs = cur_x - pre_x
            else:
                obs = cur_x
            # obs = cur_x
            # observation = cur_observation
            # print(obs)
            # observation = cur_x
            action = RL.choose_action(observation=obs)
            observation_, reward, done, _ = env.step(action)

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

        if i_episode % CHECK_POINT_STEP == 0 and i_episode > 0:
            saver_path = saver.save(RL.sess, checkpoint_dir + "model.ckpt")
            print("Model saved at %s" % saver_path)
            file = open(pkl_file, "wb")
            pkl.dump(ep_rs_sum_all, file)
            file.close()


     # plot
    file = open(pkl_file, "rb")
    ep_rs_sum_all = pkl.load(file)

    plt.plot(ep_rs_sum_all)
    plt.xlabel("episode steps")
    plt.ylabel("sum of reward in episode")
    plt.show()