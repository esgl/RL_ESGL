from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradientConvNN():
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.99, output_graph=True):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.eps_ob, self.eps_act, self.eps_re = [], [], []
        self._build_net()

        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter("logs/PolicyGradientConvNN/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, name="conv"):
        with tf.name_scope("input"):
            self.obs = tf.placeholder(tf.float32, [None, self.n_features], name="obs")
            self.act = tf.placeholder(tf.int32, [None,], name='act')
            self.re = tf.placeholder(tf.float32, [None,], name='re')

        with tf.name_scope("reshape"):
            x_image = tf.reshape(self.obs, [-1, 80, 80, 1]) # 80 * 80 * 1

        with tf.name_scope("conv1"):
            layer = tf.layers.conv2d(x_image, 10, (5, 5), activation=tf.nn.relu) # 76 * 76 * 10

        with tf.name_scope("pool1"):
            layer = tf.layers.max_pooling2d(layer, (2, 2), strides=2) # 38 * 38 * 10

        with tf.name_scope("conv2"):
            layer = tf.layers.conv2d(layer, 5, (4, 4), strides=2, activation=tf.nn.relu) # 18 * 18 * 5

        with tf.name_scope("pool2"):
            layer = tf.layers.max_pooling2d(layer, (2, 2), strides=2) # 9 * 9 * 5

        with tf.name_scope("reshape2"):
            layer = tf.reshape(layer, [-1, 9 * 9 * 5])

        with tf.name_scope("fc1"):
            layer = tf.layers.dense(
                inputs=layer,
                units=200,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
                bias_initializer=tf.constant_initializer(0.1),
                name="fc1"
            )

        with tf.name_scope("fc2"):
            all_act = tf.layers.dense(
                inputs=layer,
                units=self.n_actions,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
                bias_initializer=tf.constant_initializer(0.1),
                name="fc2"
            )

        with tf.name_scope("softmax"):
            self.all_act_prob = tf.nn.softmax(all_act, name="act_prob")

        with tf.name_scope("loss"):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.act)
            loss = tf.reduce_mean(neg_log_prob * self.re)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        # return np.random.randint(self.n_action)
        prob_weight = self.sess.run(self.all_act_prob, feed_dict={self.obs : observation[np.newaxis, :]})
        # print(prob_weight)
        action = np.random.choice(range(prob_weight.shape[1]), p=prob_weight.ravel())

        return action

    def store_transition(self, s, a, r):
        self.eps_ob.append(s)
        self.eps_act.append(a)
        self.eps_re.append(r)

    def learning(self, isNorm=True):
        if isNorm:
            discounted_reward_ep = self._discount_and_norm_reward()
        else:
            discounted_reward_ep = self._discount_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.obs : np.vstack(self.eps_ob),
            self.act : np.array(self.eps_act),
            self.re : discounted_reward_ep
        })
        self.eps_ob, self.eps_act, self.eps_re = [], [], []

        return discounted_reward_ep

    def _discount_and_norm_reward(self):
        discounted_reward_ep = np.zeros_like(self.eps_re)
        running_add = 0
        for t in reversed(range(len(self.eps_re))):
            if self.eps_re[t] != 0: running_add = 0
            running_add = running_add * self.gamma + self.eps_re[t]
            discounted_reward_ep[t] = running_add
        discounted_reward_ep -= np.mean(discounted_reward_ep)
        discounted_reward_ep /= np.std(discounted_reward_ep)
        return discounted_reward_ep

    def _discount_rewards(self):
        discounted_reward_ep = np.zeros_like(self.eps_re)
        running_add = 0
        for t in reversed(range(0, len(self.eps_re))):
            if self.eps_re[t] != 0: running_add = 0
            running_add = running_add * self.gamma + self.eps_re[t]
            discounted_reward_ep[t] = running_add
        return discounted_reward_ep


