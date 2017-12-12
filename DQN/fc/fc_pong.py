from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import gym
import numpy as np
import tensorflow as tf

from DQN.fc.DeepQNetwork import DeepQNetwork

GAME = "Pong-v0"
# RENDER = True
RENDER = False

CHECK_POING_STEP = 1000
checkpoint_dir = "logs/saver/"

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()

if __name__ == "__main__":
    episode = 1000000

    env = gym.make(GAME)
    observation = env.reset()

    actions = [2, 3]
    # n_actions = env.action_space.n
    n_actions = 2
    n_features = prepro(observation).size

    RL = DeepQNetwork(n_actions=n_actions, n_features=n_features)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(RL.sess, ckpt.model_checkpoint_path)
        print("Model restore in dir: %s" % ckpt.model_checkpoint_path)

    for i_episode in range(episode):
        cur_x = env.reset()
        cur_x = prepro(cur_x)
        pre_x = None

        ep_rs_sum = 0
        while True:
            if RENDER:
                env.render()

            if pre_x is not None:
                obs = cur_x - pre_x
            else:
                obs = cur_x

            action = RL.choose_action(obs)
            obs_, reward, done, _ = env.step(action=actions[action])
            ep_rs_sum += reward
            if reward != 0:
                text = '' if reward == -1 else "!!!!!!!!!!!!!!!!!!"
                print("episode %d  reward %d, %s" % (i_episode, reward, text))

            obs_ = prepro(obs_)
            RL.store_transition(s=cur_x, a=action, r=reward, s_=obs_)
            if done:
                print("\n***********************************************")
                print("eposide %d, return %d" % (i_episode, ep_rs_sum))
                print("***********************************************\n")
                vt = RL.learn()
                break
            pre_x = cur_x
            cur_x = obs_
        if i_episode % CHECK_POING_STEP == 0 and i_episode > 0:
            saver_path = saver.save(RL.sess, checkpoint_dir + "model.ckpt")
            print("Model saved at %s" % saver_path)

    RL.plot_cost()