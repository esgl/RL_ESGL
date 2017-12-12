from __future__ import print_function
from __future__ import division


import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient():
    def __init__(self, n_action, n_features, learning_rate=0.01, reward_decay=0.99, output_graph=True):
        self.n_action = n_action
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.ep_observations, self.ep_actions, self.ep_rewards = [], [], []
        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, name='fullconnection'):
        with tf.name_scope('input'):
            self.observation = tf.placeholder(tf.float32, [None, self.n_features], name="observation")
            self.action = tf.placeholder(tf.int32, [None, ], name="action")
            self.action_return = tf.placeholder(tf.float32, [None, ], name="action_return")
        if name == 'fullconnection':
            with tf.name_scope('fc1'):
                layer = tf.layers.dense(
                    inputs=self.observation,
                    units=10,
                    activation=tf.nn.relu,
                    use_bias=True,
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
                    # bias_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='fc1'
                )
            with tf.name_scope('fc2'):
                all_act = tf.layers.dense(
                    inputs=layer,
                    units=self.n_action,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
                    # bias_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='fc2'
                )
            self.all_act_prob = tf.nn.softmax(all_act, name="act_prob")

            with tf.name_scope("loss"):
                neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.action)
                loss = tf.reduce_mean(neg_log_prob * self.action_return)

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


    def choose_action(self, observation):
        # return np.random.randint(self.n_action)
        prob_weight = self.sess.run(self.all_act_prob, feed_dict={self.observation : observation[np.newaxis, :]})
        # print(prob_weight)
        action = np.random.choice(range(prob_weight.shape[1]), p=prob_weight.ravel())

        return action

    def store_transition(self, observation, action, reward):
        self.ep_observations.append(observation)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)

    def learning(self, isNorm=True):
        if isNorm:
            discounted_reward_ep = self._discount_and_norm_reward()
        else:
            discounted_reward_ep = self._discount_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.observation : np.vstack(self.ep_observations),
            self.action : np.array(self.ep_actions),
            self.action_return : discounted_reward_ep
        })
        self.ep_observations, self.ep_actions, self.ep_rewards = [], [], []

        return discounted_reward_ep

    def _discount_and_norm_reward(self):
        discounted_reward_ep = np.zeros_like(self.ep_rewards)
        running_add = 0
        for t in reversed(range(len(self.ep_rewards))):
            running_add = running_add * self.reward_decay + self.ep_rewards[t]
            discounted_reward_ep[t] = running_add
        discounted_reward_ep -= np.mean(discounted_reward_ep)
        discounted_reward_ep /= np.std(discounted_reward_ep)
        return discounted_reward_ep

    def _discount_rewards(self):
        discounted_reward_ep = np.zeros_like(self.ep_rewards)
        running_add = 0
        for t in reversed(range(len(self.ep_rewards))):
            running_add = running_add * self.reward_decay + self.ep_rewards[t]
            discounted_reward_ep[t] = running_add
        return discounted_reward_ep
