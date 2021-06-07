# MIT License

# Copyright (c) 2017

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from bisect import bisect

import numpy as np
from scipy.special import softmax
from sortedcontainers import SortedDict


# Adapted from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
# The MIT license was taken from there
class TabularRLAlgorithm:

   AVAILABLE_POLICIES = ['egreedy', 'boltzmann']

   def __init__(self, actions, policy='egreedy', learning_rate=0.01, reward_decay=0.9, epsilon=0.1):
      """Implementation of a tabular RL algorithm in Python

      :param actions: number of actions
      :type actions: int
      :param policy: name of one of the available policies, defaults to 'egreedy'
      :type policy: str, optional
      :param learning_rate: learning rate of the algorithm, defaults to 0.01
      :type learning_rate: float, optional
      :param reward_decay: reward decay, defaults to 0.9
      :type reward_decay: float, optional
      :param epsilon: probability of taking a random action in the e-greedy policy, defaults to 0.1
      :type epsilon: float, optional
      :raises ValueError: if an unknown policy name is passed as argument
      :return: an object which implements functions to update the Q-table, as well as select actions according to policies and the values in the Q-table
      :rtype: QLearningTable
      """
      if policy not in TabularRLAlgorithm.AVAILABLE_POLICIES:
         raise ValueError('Unknown policy \'{}\'. Choose one from {}'.format(
             policy, TabularRLAlgorithm.AVAILABLE_POLICIES))

      self.actions = actions
      self.policy = policy
      self.lr = learning_rate
      self.gamma = reward_decay
      self.epsilon = epsilon
      self.q_table = SortedDict()

   def _boltzmann_policy(self, s: str) -> int:
      """Select an action for the given state `s` according to a Boltzmann policy, in which the probabilities of each action being chosen is equal to their softmaxed values

      :param s: a state
      :type s: str
      :return: chosen action
      """
      # https://stackoverflow.com/a/4442687/1245214
      # create a cdf of the softmaxed values and find where a
      # number between 0 and 1 would be inserted in the cdf list
      softmaxed_q_values_cdf = softmax(self.q_table[s]).cumsum()
      return bisect(softmaxed_q_values_cdf, np.random.uniform())

   def _egreedy_policy(self, s: str) -> int:
      """Randomly select an action with probability epsilon, or select the best action for the given state `s` with probability 1 - epsilon

      :param s: a state
      :type s: str
      :return: chosen action
      """
      if np.random.uniform() > self.epsilon:
         state_actions = self.q_table[s]

         # get actions with largest value
         best_actions = np.nonzero(state_actions == state_actions.max())[0]

         # some actions may have the same value
         action = np.random.choice(best_actions)
      else:
         # choose random action
         action = np.random.choice(self.actions)

      return action

   def choose_action(self, s: str) -> int:
      self._check_state_exist(s)

      if self.policy == 'egreedy':
         action = self._egreedy_policy(s)
      elif self.policy == 'boltzmann':
         action = self._boltzmann_policy(s)

      return action

   def _check_state_exist(self, s: str):
      if s not in self.q_table:
         self.q_table[s] = np.zeros(self.actions)

   def learn(self, s: str, a: int, r: float, s_: str):
      """Update the Q-table

      :param s: current state
      :param a: action taken
      :param r: reward signal
      :param s_: observed future state
      """
      pass


class QLearning(TabularRLAlgorithm):

   def learn(self, s: str, a: int, r: float, s_: str):
      self._check_state_exist(s_)
      self._check_state_exist(s)

      q_predict = self.q_table[s][a]
      q_target = r + self.gamma * np.max(self.q_table[s_])

      # update
      self.q_table[s][a] += self.lr * (q_target - q_predict)


class Sarsa(TabularRLAlgorithm):

   def learn(self, s: str, a: int, r: float, s_: str):
      self._check_state_exist(s_)
      self._check_state_exist(s)

      q_predict = self.q_table[s][a]
      q_target = r + self.gamma * self.q_table[s_][a]

      # update
      self.q_table[s][a] += self.lr * (q_target - q_predict)


class TabularRLLambdaAlgorithm(TabularRLAlgorithm):

   def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon=0.1, trace_decay=0.9):
      super(TabularRLAlgorithm, self).__init__(actions, learning_rate, reward_decay, epsilon)

      # backward view, eligibility trace.
      self.lambda_ = trace_decay
      self.eligibility_trace = self.q_table.copy()

   def _check_state_exist(self, s):
      if s not in self.q_table:
         self.q_table[s] = np.zeros(self.actions)
         self.eligibility_trace[s] = np.zeros(self.actions)

   def update_trace(self, s, a, error):
      # Method 1:
      # self.eligibility_trace[s][a] += 1

      # Method 2:
      self.eligibility_trace[s] = np.zeros(self.actions)
      self.eligibility_trace[s][a] = 1

      # Q update
      self.q_table += self.lr * error * self.eligibility_trace

      # decay eligibility trace after update
      self.eligibility_trace *= self.gamma * self.lambda_


class QLambda(TabularRLLambdaAlgorithm):

   def learn(self, s, a, r, s_, a_):
      self._check_state_exist(s_)
      self._check_state_exist(s)
      q_predict = self.q_table[s][a]
      if s_ != 'terminal':
         q_target = r + self.gamma * np.max(self.q_table[s_])  # next state is not terminal
      else:
         q_target = r  # next state is terminal
      error = q_target - q_predict

      # increase trace amount for visited state-action pair
      self.update_trace(s, a, error)


class SarsaLambda(TabularRLLambdaAlgorithm):

   def learn(self, s, a, r, s_, a_):
      self._check_state_exist(s_)
      self._check_state_exist(s)
      q_predict = self.q_table[s][a]
      if s_ != 'terminal':
         q_target = r + self.gamma * self.q_table[s_][a_]  # next state is not terminal
      else:
         q_target = r  # next state is terminal
      error = q_target - q_predict

      # increase trace amount for visited state-action pair
      self.update_trace(s, a, error)
