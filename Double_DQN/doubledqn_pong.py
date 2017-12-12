from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import gym
from Double_DQN.DoubleDQN import DoubleDQN
from util.prepro import prepro
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make("Pong-v0")
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTIONS = [2, 3]
ACTION_SPACE = 2

RENDER = False
# RENDER = True

CHECK_POINT_STEP = 1000
checkpoint_dir = "logs/saver/"
observation = env.reset()
observation = prepro(observation)
n_features = observation.size

sess = tf.Session()
with tf.variable_scope("Natural_DQN"):
    nature_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=n_features, memory_size=MEMORY_SIZE, e_greedy_increment=0.001,
        double_q=False, sess=sess)
with tf.variable_scope("Double_DQN"):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=n_features, memory_size=MEMORY_SIZE, e_greedy_increment=0.001,
        double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())

def train(RL):
    total_steps = 0
    observation = env.reset()
    observation = prepro(observation)
    while True:
        if RENDER:
            env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(ACTIONS[action])

        # reward /= 10
        if reward == 1:
            print("reward ++++++++++++++++++++++++")
        elif reward == -1:
            print("reward ", reward)

        observation_ = prepro(observation_)
        RL.store_transition(observation, action, reward, observation_)
        if total_steps > MEMORY_SIZE:
            RL.learn()
        if total_steps - MEMORY_SIZE > 10000:
            break

        if total_steps % 1000 == 0:
            print("steps %d" % total_steps)

        observation = observation_
        total_steps += 1
    return RL.q

q_natural = train(nature_DQN)
q_double = train(double_DQN)

plt.plot(np.array(q_natural), c='r', label="natural")
plt.plot(np.array(q_double), c='b', label="double")
plt.legend(loc="best")
plt.xlabel("training steps")
plt.xlabel("Q eval")
plt.grid()
plt.show()
