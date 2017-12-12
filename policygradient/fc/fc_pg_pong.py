from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import gym
import numpy as np
# import PIL.Image as Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
import os



from policygradient.fc.PolicyGradient import PolicyGradient


GAME = 'Pong-v0'
# RENDER = True
RENDER = False
DISPLAY_REWARD_THRESHOLD = -1

env = gym.make(GAME)
print(env.action_space)
print(env.action_space.n)
print(env.observation_space)
print(env.observation_space.shape[0])

CHECK_POINT_STEP = 1000
checkpoint_dir = "logs/saver/"

pkl_file = "logs/saver/ep_rs_sum_all.pkl"

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()

if __name__ == '__main__':

    episode = 1000000
    observation_tmp = env.reset()
    observation_tmp = prepro(observation_tmp)
    n_observation_space = observation_tmp.size
    print('n_observation_space', n_observation_space)

    RL = PolicyGradient(n_features=n_observation_space)
    ep_rs_sum_all = []

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(RL.sess, ckpt.model_checkpoint_path)
        print("Model restore in file: %s" % ckpt.model_checkpoint_path)

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
            if pre_observation is not None:
                observation = cur_observation - pre_observation
            else:
                observation = cur_observation
            # print(observation)

            action = RL.choose_action(observation)
            # print('action: ', action)
            observation_, reward, done, _ = env.step(action)
            # print('reward: ', reward)

            if reward != 0:
                score += 1
                text = '' if reward == -1 else '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                print('eposide %d  reward %d, %s' % (i_episode, reward, text))

            observation_ = prepro(observation_)

            RL.store_transition(s=observation_, a=action, r=reward)
            if done:
                ep_rs_sum = sum(RL.eps_re)
                ep_rs_sum_all.append(ep_rs_sum)
                print("\n***********************************************")
                print("eposide %d, return %d" % (i_episode, ep_rs_sum))
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
        if i_episode % CHECK_POINT_STEP == 0:
            saver_path = saver.save(RL.sess, checkpoint_dir + "model.ckpt")
            print("Model saved at %s" % saver_path)

            # if os.path.exists(pkl_file):
            #     os.remove(pkl_file)
            file = open(pkl_file, "wb")
            pkl.dump(ep_rs_sum_all, file)
            file.close()

    file = open(pkl_file, "rb")
    ep_rs_sum_all = pkl.load(file)

    plt.plot(ep_rs_sum_all)
    plt.xlabel("episode steps")
    plt.ylabel("sum of reward in episode")
    plt.show()
