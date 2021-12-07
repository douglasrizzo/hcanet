import random
from collections import namedtuple

from .segment_tree import MinSegmentTree, SumSegmentTree

# the original algorithm used a namedtuple, but those can't be documented.
# I learned a hack on how to document them here https://stackoverflow.com/a/1606478
Transition_ = namedtuple("Transition", ("state", "action", "reward", "done", "av_actions"))


class Transition(Transition_):
   """A named tuple to store a state transition (s, a, s', r)"""

   pass


class ReplayBuffer:
   def __init__(self, size):
      """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
      self._storage = []
      self._maxsize = size
      self._next_idx = 0
      self.is_prioritized = False

   def __len__(self):
      return len(self._storage)

   def can_sample(self, batch_size):
      return len(self._storage) >= batch_size

   def add(self, obs_t, action, reward, obs_tp1, done):
      data = Transition(obs_t, action, reward, obs_tp1, done)

      if self._next_idx >= len(self._storage):
         self._storage.append(data)
      else:
         self._storage[self._next_idx] = data
      self._next_idx = (self._next_idx + 1) % self._maxsize

   def extend(self, obs_t, action, reward, obs_tp1, done):
      """
        add a new batch of transitions to the buffer
        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch
        Note: uses the same names as .add to keep compatibility with named argument passing
                 but expects iterables and arrays with more than 1 dimensions
        """
      for data in zip(obs_t, action, reward, obs_tp1, done):
         if self._next_idx >= len(self._storage):
            self._storage.append(data)
         else:
            self._storage[self._next_idx] = data
         self._next_idx = (self._next_idx + 1) % self._maxsize

   def sample(self, batch_size):
      """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        transitions: [Transition]
            batch of transitions
        """
      return random.sample(self._storage, batch_size)


class PrioritizedReplayBuffer(ReplayBuffer):
   def __init__(self, size, alpha=0.6):
      """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
      super().__init__(size)
      assert alpha >= 0
      self._alpha = alpha

      it_capacity = 1
      while it_capacity < size:
         it_capacity *= 2

      self._it_sum = SumSegmentTree(it_capacity)
      self._it_min = MinSegmentTree(it_capacity)

      self._max_priority = 1.0
      self.is_prioritized = True

   def add(self, obs_t, action, reward, obs_tp1, done):
      """See ReplayBuffer.store_effect"""
      idx = self._next_idx
      super().add(obs_t, action, reward, obs_tp1, done)
      self._it_sum[idx] = self._max_priority**self._alpha
      self._it_min[idx] = self._max_priority**self._alpha

   def extend(self, obs_t, action, reward, obs_tp1, done):
      """
        add a new batch of transitions to the buffer
        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch
        Note: uses the same names as .add to keep compatibility with named argument passing
           but expects iterables and arrays with more than 1 dimensions
        """
      idx = self._next_idx
      super().extend(obs_t, action, reward, obs_tp1, done)
      while idx != self._next_idx:
         self._it_sum[idx] = self._max_priority**self._alpha
         self._it_min[idx] = self._max_priority**self._alpha
         idx = (idx + 1) % self._maxsize

   def _sample_proportional(self, batch_size):
      res = []
      p_total = self._it_sum.sum(0, len(self._storage) - 1)
      every_range_len = p_total / batch_size
      for i in range(batch_size):
         mass = random.random() * every_range_len + i * every_range_len
         idx = self._it_sum.find_prefixsum_idx(mass)
         res.append(idx)
      return res

   def sample(self, batch_size, beta=0.4):
      """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        transitions: [Transition]
            batch of transitions
        weights: [float]
            List of size (batch_size) and dtype float
            denoting importance weight of each sampled transition
        idxes: [int]
            List of size (batch_size) and dtype int
            indexes in buffer of sampled experiences
        """
      assert beta > 0

      idxes = self._sample_proportional(batch_size)

      weights = []
      p_min = self._it_min.min() / self._it_sum.sum()
      max_weight = (p_min * len(self._storage))**(-beta)

      for idx in idxes:
         p_sample = self._it_sum[idx] / self._it_sum.sum()
         weight = (p_sample * len(self._storage))**(-beta)
         weights.append(weight / max_weight)

      transitions = [self._storage[idx] for idx in idxes]
      return transitions, weights, idxes

   def update_priorities(self, idxes, priorities):
      """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
      assert len(idxes) == len(priorities)
      assert all(p > 0 for p in priorities)
      assert all(0 <= idx < len(self._storage) for idx in idxes)

      for idx, priority in zip(idxes, priorities):
         self._it_sum[idx] = priority**self._alpha
         self._it_min[idx] = priority**self._alpha

      self._max_priority = max([self._max_priority] + priorities)

   def copy(self, sample_size: int):
      """
        :param sample_size: number of items to include in the new object
        :type sample_size: int
        :return: a new replay buffer, retaining
        :rtype: PrioritizedReplayBuffer
        """
      newb = PrioritizedReplayBuffer(self._maxsize, self._alpha)
      transitions, weights, idxes = self.sample(sample_size)

      newb._storage = transitions
      newb._next_idx = len(transitions)

      for i in range(sample_size):
         newb._it_sum[i] = self._it_sum[idxes[i]]
         newb._it_min[i] = self._it_min[idxes[i]]

         newb._max_priority = max(self._max_priority, newb._it_sum[i])

      return newb


class EpisodeReplayBuffer:
   def __init__(self, size):
      """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of episodes to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
      self._storage = []
      self._maxsize = size
      self._next_idx = 0
      self.is_prioritized = False

   def __len__(self):
      return len(self._storage)

   def can_sample(self, batch_size):
      return len(self._storage) >= batch_size

   def add(self, data: list):
      if self._next_idx >= len(self._storage):
         self._storage.append(data)
      else:
         self._storage[self._next_idx] = data
      self._next_idx = (self._next_idx + 1) % self._maxsize

   def sample(self, batch_size):
      """Sample a sequential batch of episodes.
        Parameters
        ----------
        batch_size: int
            How many episodes to sample.
        Returns
        -------
        transitions: list
            batch of episodes
        """
      return random.sample(self._storage, batch_size)

   def extend(self, datas):
      """
        add a new batch of episodes to the buffer
        :param datas: a list of transitions
        """
      for data in datas:
         if self._next_idx >= len(self._storage):
            self._storage.append(data)
         else:
            self._storage[self._next_idx] = data
         self._next_idx = (self._next_idx + 1) % self._maxsize


class PrioritizedEpisodeReplayBuffer(EpisodeReplayBuffer):
   def __init__(self, size, alpha=0.6):
      """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of episodes to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        EpisodeReplayBuffer.__init__
        """
      super().__init__(size)
      assert alpha >= 0
      self._alpha = alpha

      it_capacity = 1
      while it_capacity < size:
         it_capacity *= 2

      self._it_sum = SumSegmentTree(it_capacity)
      self._it_min = MinSegmentTree(it_capacity)

      self._max_priority = 1.0
      self.is_prioritized = True

   def add(self, data):
      """See ReplayBuffer.store_effect"""
      idx = self._next_idx
      super().add(data)
      self._it_sum[idx] = self._max_priority**self._alpha
      self._it_min[idx] = self._max_priority**self._alpha

   def extend(self, datas):
      """
        add a new batch of episodes to the buffer
        :param datas: a list of transitions
        """
      idx = self._next_idx
      super().extend(datas)
      while idx != self._next_idx:
         self._it_sum[idx] = self._max_priority**self._alpha
         self._it_min[idx] = self._max_priority**self._alpha
         idx = (idx + 1) % self._maxsize

   def _sample_proportional(self, batch_size):
      res = []
      p_total = self._it_sum.sum(0, len(self._storage) - 1)
      every_range_len = p_total / batch_size
      for i in range(batch_size):
         mass = random.random() * every_range_len + i * every_range_len
         idx = self._it_sum.find_prefixsum_idx(mass)
         res.append(idx)
      return res

   def sample(self, batch_size, beta=0.4):
      """Sample a batch of experiences.
        compared to EpisodeReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        transitions: list
            batch of episodes
        weights: [float]
            List of size (batch_size) and dtype float
            denoting importance weight of each sampled episode
        idxes: [int]
            List of size (batch_size) and dtype int
            indexes in buffer of sampled experiences
        """
      assert beta > 0

      idxes = self._sample_proportional(batch_size)

      weights = []
      p_min = self._it_min.min() / self._it_sum.sum()
      max_weight = (p_min * len(self._storage))**(-beta)

      for idx in idxes:
         p_sample = self._it_sum[idx] / self._it_sum.sum()
         weight = (p_sample * len(self._storage))**(-beta)
         weights.append(weight / max_weight)

      transitions = [self._storage[idx] for idx in idxes]
      return transitions, weights, idxes

   def update_priorities(self, idxes, priorities):
      """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
      assert len(idxes) == len(priorities)
      assert all(p > 0 for p in priorities)
      assert all(0 <= idx < len(self._storage) for idx in idxes)

      for idx, priority in zip(idxes, priorities):
         self._it_sum[idx] = priority**self._alpha
         self._it_min[idx] = priority**self._alpha

      self._max_priority = max([self._max_priority] + priorities)

   def copy(self, sample_size: int):
      """
        :param sample_size: number of items to include in the new object
        :type sample_size: int
        :return: a new replay buffer, retaining
        :rtype: PrioritizedReplayBuffer
        """
      newb = PrioritizedEpisodeReplayBuffer(self._maxsize, self._alpha)
      transitions, weights, idxes = self.sample(sample_size)

      newb._storage = transitions
      newb._next_idx = len(transitions)

      for i in range(sample_size):
         newb._it_sum[i] = self._it_sum[idxes[i]]
         newb._it_min[i] = self._it_min[idxes[i]]

         newb._max_priority = max(self._max_priority, newb._it_sum[i])

      return newb
