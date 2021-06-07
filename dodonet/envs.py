import numpy as np
import torch
from smac.env import StarCraft2Env
from torch_geometric.data import Data

from dodonet.encoding import isin
# from pommerman.envs.v0 import Pomme


class GraphEnv:

    def __init__(self, device):
        self.device = device

    def add_all_agent_edges(self, state: Data, agent_types: torch.Tensor) -> Data:
        """Add edges between all agent nodes

        :param state: a graph state
        :type state: Data
        :return: the same graph state, with edges between all agents added
        :rtype: Data
        """
        agent_nodes = isin(state.node_type, agent_types)
        agent_edge_source: list = []
        agent_edge_target: list = []

        for source in agent_nodes:
            agent_edge_source += [source] * agent_nodes.size(0)
            agent_edge_target += agent_nodes

        agent_edge_index = torch.tensor(
            [agent_edge_source, agent_edge_target], dtype=torch.long, device=self.device
        )

        state.edge_index = torch.cat((state.edge_index, agent_edge_index), dim=-1)

        return state.coalesce()

    def get_graph_state(self, *args):
        raise NotImplementedError


class SMACEnv(GraphEnv, StarCraft2Env):

    def __init__(self, map_name, replay_dir, reward_sparse, device):
        StarCraft2Env.__init__(
            self, map_name=map_name, replay_dir=replay_dir, reward_sparse=reward_sparse
        )
        GraphEnv.__init__(self, device)

    @staticmethod
    def _visibility_matrix_to_edge_index(adj: np.ndarray):
        """Transforms a dense adjacency matrix, representing whether agent nodes view other nodes,
        into a [2, n] PyTorch Geometric edge_index tensor, where the first row represents source
        nodes (the nodes seen be the agents) and the second tensor represent target nodes (the
        agent nodes themselves).

        :param adj: an agents x nodes adjacency matrix
        :type adj: numpy.ndarray
        :return: a [2, n] PyTorch Geometric edge_index tensor
        :rtype: torch.Tensor
        """
        coo = torch.zeros((2, np.sum(adj > 0)), dtype=torch.long)

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

    def get_graph_state(self, node_types: torch.Tensor, agent_types: torch.Tensor = None) -> Data:
        """Transforms the current SMAC state into a PyTorch Geometric graph

        :param node_types: tensor containing unit types
        :type node_types: torch.Tensor
        :return: a graph in which agent nodes are connected to the nodes that are visible to them and each node contains the state observation wrt. to that agent/node. Zero padding is used to make all observation vectors have the same size.
        :rtype: torch_geometric.data.Data
        """
        state_dict = self.get_state_dict()
        visibility_matrix = self.get_visibility_matrix()

        # separate allies from enemies
        ally_state = state_dict["allies"]
        enemy_state = state_dict["enemies"]

        # remove unit type from observation, as it is encoded in the network itself
        # TODO is it?
        if self.unit_type_bits > 0:
            ally_state = ally_state[:, :-self.unit_type_bits]
            enemy_state = enemy_state[:, :-self.unit_type_bits]

        # join observation vectors and transform into a single tensor
        x = SMACEnv._pad_concat([ally_state, enemy_state]).astype(np.float32)
        x = torch.tensor(x, device=self.device)

        edge_index = SMACEnv._visibility_matrix_to_edge_index(visibility_matrix).to(self.device)

        state = Data(x=x, edge_index=edge_index, node_type=node_types)

        if agent_types is not None:
            state = super().add_all_agent_edges(state, agent_types)

        return state


# class PommermanEnv(GraphEnv, Pomme):

#     @staticmethod
#     def get_feacture_vector_len():
#         return 614

#     def get_graph_state(self, s: dict, my_players: list, agent_types:torch.Tensor=None):
#         # TODO ver o dicionário do estado do ambiente e pensar em como transformar em grafo
#         # precisa ter x, edge_index e node_types
#         # alive position blast_strength int(can_kick) ammo
#         # 'board', 'bomb_blast_strength', 'bomb_life', 'bomb_moving_direction', 'flame_life'
#         x = torch.zeros(
#             (len(s), PommermanEnv.get_feacture_vector_len()), dtype=torch.float, device=self.device
#         )

#         # TODO maybe this is unnecessary, since it's encoded in the graph
#         # it's also present in all node vectors... mmm...
#         alive = torch.zeros(4, dtype=torch.float)
#         for i in s[0]["alive"]:
#             alive[i - 10] = 1

#         for i, obs in enumerate(s):
#             # these are lists, single values and tuples
#             position = torch.tensor(obs["position"], dtype=torch.float)  # 2
#             blast_strength = torch.tensor(obs["blast_strength"],
#                                           dtype=torch.float).unsqueeze(0)  # 1
#             can_kick = torch.tensor(obs["can_kick"], dtype=torch.float).unsqueeze(0)  # 1
#             ammo = torch.tensor(obs["ammo"], dtype=torch.float).unsqueeze(0)  # 1
#             board = torch.tensor(obs["board"].flatten(), dtype=torch.float)  # 11x11
#             bomb_blast_strength = torch.tensor(
#                 obs["bomb_blast_strength"].flatten(), dtype=torch.float
#             )  # 11x11
#             bomb_life = torch.tensor(obs["bomb_life"].flatten(), dtype=torch.float)  # 11x11
#             bomb_moving_direction = torch.tensor(
#                 obs["bomb_moving_direction"].flatten(), dtype=torch.float
#             )  # 11x11
#             flame_life = torch.tensor(obs["flame_life"].flatten(), dtype=torch.float)  # 11x11
#             mini_x = torch.cat(
#                 (
#                     alive,
#                     position,
#                     blast_strength,
#                     can_kick,
#                     ammo,
#                     board,
#                     bomb_blast_strength,
#                     bomb_life,
#                     bomb_moving_direction,
#                     flame_life,
#                 )
#             )  # 614
#             x[i] = mini_x

#         source, target = [], []
#         node_types = torch.ones(4, dtype=torch.int)

#         for player in my_players:
#             me = player + 10
#             node_types[player] = 0

#             if me in s[0]["alive"]:
#                 obs = s[player]

#                 for j in obs["alive"]:
#                     if j != me and j in obs["board"]:
#                         source.append(j - 10)
#                         target.append(player)

#         edge_index = torch.stack(
#             (torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long))
#         )

#         state = Data(x=x, edge_index=edge_index, node_type=node_types)
#         if agent_types is not None:
#             state = super().add_all_agent_edges(agent_types)

#         return state

#     def act(self, output: Data, step_num: int) -> torch.Tensor:
#         valid_actions = [[True] * act for act in self.n_actions_agents]
#         return super().act(self.policy_net, output, valid_actions, step_num)
