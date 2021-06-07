import numpy as np
import torch as th
from smac.env import StarCraft2Env
from torch_geometric.data import Data

from hcanet.encoding import isin


class GraphEnv:

   def add_all_agent_edges(self, state: Data, agent_types: th.tensor,
                           node_types: th.tensor) -> Data:
      """Add edges between all agent nodes

      :param state: a graph state
      :type state: Data
      :return: the same graph state, with edges between all agents added
      :rtype: Data
      """
      agent_nodes = isin(node_types, agent_types)
      agent_edge_source: list = []
      agent_edge_target: list = []

      for source in agent_nodes:
         agent_edge_source += [source] * agent_nodes.size(0)
         agent_edge_target += agent_nodes

      agent_edge_index = th.tensor([agent_edge_source, agent_edge_target], dtype=th.long)

      state.edge_index = th.cat((state.edge_index, agent_edge_index), dim=-1)

      return state.coalesce()

   def get_graph_state(self, *args):
      raise NotImplementedError


class SMACEnv(GraphEnv, StarCraft2Env):

   def __init__(self,
                map_name,
                replay_dir,
                reward_sparse=False,
                reward_only_positive=True,
                reward_death_value=10,
                reward_win=200,
                reward_defeat=0,
                reward_negative_scale=0.5,
                reward_scale=True,
                reward_scale_rate=20,
                append_unit_ids: bool = True):
      StarCraft2Env.__init__(self,
                             map_name=map_name,
                             replay_dir=replay_dir,
                             reward_sparse=reward_sparse,
                             reward_only_positive=reward_only_positive,
                             reward_death_value=reward_death_value,
                             reward_win=reward_win,
                             reward_defeat=reward_defeat,
                             reward_negative_scale=reward_negative_scale,
                             reward_scale=reward_scale,
                             reward_scale_rate=reward_scale_rate)
      GraphEnv.__init__(self)
      self.include_unit_ids = append_unit_ids

   @staticmethod
   def _visibility_matrix_to_edge_index(adj: np.ndarray):
      """Transforms a dense adjacency matrix, representing whether agent nodes view other nodes,
      into a [2, n] PyTorch Geometric edge_index tensor, where the first row represents source
      nodes (the nodes seen be the agents) and the second tensor represent target nodes (the
      agent nodes themselves).

      :param adj: an agents x nodes adjacency matrix
      :type adj: numpy.ndarray
      :return: a [2, n] PyTorch Geometric edge_index tensor
      :rtype: th.tensor
      """
      coo = th.zeros((2, np.sum(adj > 0)), dtype=th.long)

      # in this bit, x represents agent nodes, y represents all nodes and value > 0 indicates that agent x sees node y
      i = 0
      for (x, y), value in np.ndenumerate(adj):
         if value > 0:
            # PyG expects the source nodes (which will send messages) to be in the first row of the tensor
            # and target tensors (which will aggregate messages received by source nodes) in the second row.
            # since agents are the ones aggregating messages, they are kept in row 1 and the other node in row 0
            coo[0, i] = y
            coo[1, i] = x
            i += 1

      return coo

   @staticmethod
   def _pad_concat(arrs: list) -> np.ndarray:
      """Pads a list of numpy.ndarray objects so that all arrays have the same number of columns and then stacks them into a single 2D array.

      :param arrs: list of numpy.ndarray objects
      :type arrs: list(numpy.ndarray)
      :return: numpy array, in which the original arrays have been stacked and padded with zeros
      :rtype: numpy.ndarray
      """
      n_cols = max([arr.shape[1] for arr in arrs])

      for i in range(len(arrs)):
         arrs[i] = np.pad(arrs[i], ((0, 0), (0, n_cols - arrs[i].shape[1])))

      return np.vstack(arrs)

   def get_graph_state(self,
                       node_types: th.tensor = None,
                       agent_types: th.tensor = None,
                       include_unit_ids: bool = None,
                       include_unit_types: bool = False,
                       v2: bool = True) -> Data:
      """Transforms the current SMAC state into a PyTorch Geometric graph

      :param node_types: tensor containing unit types
      :type node_types: th.tensor
      :return: a graph in which agent nodes are connected to the nodes that are visible to them and each node contains the state observation wrt. to that agent/node. Zero padding is used to make all observation vectors have the same size.
      :rtype: torch_geometric.data.Data
      """
      if include_unit_ids is None:
         include_unit_ids = self.include_unit_ids

      visibility_matrix = self.get_visibility_matrix()
      if not v2:
         state_dict = self.get_state_dict()
         unit_states = list()

         # separate allies from enemies
         for unit_state in state_dict["allies"], state_dict["enemies"]:
            u = unit_state

            # remove unit type from observation, as it is encoded in the network itself
            if not include_unit_types and self.unit_type_bits > 0:
               u = u[:, :-self.unit_type_bits]
            if include_unit_ids:
               u = np.concatenate((np.eye(len(u)), u), axis=1)

            unit_states.append(u)

         if "last_action" in state_dict:
            unit_states[0] = np.concatenate((unit_states[0], state_dict["last_action"]), axis=1)

         # join observation vectors and transform into a single tensor
         x = SMACEnv._pad_concat(unit_states).astype(np.float32)
         x = th.tensor(x)
      else:
         x = th.tensor(self.get_obs())
         if include_unit_ids:
            x = th.cat((th.eye(x.size(0)), x), dim=1)
         visibility_matrix = visibility_matrix[:, :visibility_matrix.shape[0]]

      edge_index = SMACEnv._visibility_matrix_to_edge_index(visibility_matrix)

      state = Data(x=x, edge_index=edge_index)

      if agent_types is not None:
         state = super().add_all_agent_edges(state, agent_types, node_types)

      return state

   def get_obs_size(self):
      obs_size = super().get_obs_size()
      return obs_size if not self.include_unit_ids else obs_size + self.n_agents
