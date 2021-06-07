from enum import Enum

import torch
from torch_geometric.data import Data

from dodonet.encoding import LabelEncoder, isin

from .modules.action import (
    HeteroMAGNetActionLayer, MultiAgentMultiNetActorCriticLayer, MultiAgentMultiNetQLayer
)
from .modules.encoding import EncoderByType
from .modules.graph import GATModule, GCNModule, GraphModule, RGCNModule


class HeteroMAGNetBaseLayer(torch.nn.Module):

    class GraphLayerType(Enum):
        GCN = 'GCN'
        GAT = 'GAT'
        RGCN = 'RGCN'

    def __init__(
        self,
        num_node_types: int,
        agent_node_types: list,
        node_type_input_sizes: list,
        encoding_hidden_sizes: list,
        encoding_output_size: int,
        relational_hidden_sizes: list,
        relational_output_size: int,
        device,
        graph_layer_type: GraphLayerType = GraphLayerType.GCN,
        full_receptive_field: bool = True,
        gat_n_heads: int = 1,
        gat_average_last: bool = False,
        rgcn_n2_relations: bool = True,
        rgcn_num_bases: int = 1
    ):
        """This class encapsulates the encoding and graph layers of HeteroMAGNet

        :param num_node_types: Number of node types
        :type num_node_types: int
        :param agent_node_types: List of which node types are agents
        :type agent_node_types: list
        :param node_type_input_sizes: Size of input vectors for each node type
        :type node_type_input_sizes: list
        :param encoding_hidden_sizes: List of number of neurons in the hidden layers of the encoding networks
        :type encoding_hidden_sizes: list
        :param encoding_output_size: Size of the output for the encoding networks
        :type encoding_output_size: int
        :param relational_hidden_sizes: List of number of neurons in the graph convolutional layers
        :type relational_hidden_sizes: int
        :param relational_output_size: Size of the output for the final graph convolutional layer
        :type relational_output_size: int
        :param device: PyTorch device
        :param graph_layer_type: which graph layer to use, defaults to GraphLayerType.GCN
        :type graph_layer_type: GraphLayerType, optional
        :param full_receptive_field: whether to return a concatenation of the outputs of all graph layers, defaults to True
        :type full_receptive_field: bool, optional
        :param gat_n_heads: number of heads for each GAT layer in a GAT module, defaults to 1
        :type gat_n_heads: int, optional
        :param gat_average_last: whether to average the heads of the last GAT, defaults to False (concatenation)
        :type gat_average_last: bool, optional
        :param rgcn_n2_relations: ``True``: generate one relation between each agent-node connection. ``False`` generate one relation for each agent node type, to be used against every other node, regardless of type, defaults to True
        :type rgcn_n2_relations: bool, optional
        :param rgcn_num_bases: number of basis matrices that compose each individual RGCN layer, defaults to 1
        :type rgcn_num_bases: int, optional
        """
        super().__init__()

        self.device = device
        self.node_type_input_sizes = node_type_input_sizes
        self.graph_layer_type = graph_layer_type

        self.encoding_layer = EncoderByType(
            node_type_input_sizes, encoding_hidden_sizes, encoding_output_size, device
        )

        if graph_layer_type == HeteroMAGNetBaseLayer.GraphLayerType.GCN:
            self.relational_layer = GCNModule(
                self.encoding_layer.out_features, relational_hidden_sizes, relational_output_size,
                num_node_types, agent_node_types, device, full_receptive_field
            )  # type: GraphModule
        elif graph_layer_type == HeteroMAGNetBaseLayer.GraphLayerType.GAT:
            self.relational_layer = GATModule(
                self.encoding_layer.out_features, relational_hidden_sizes, relational_output_size,
                num_node_types, agent_node_types, device, full_receptive_field, gat_n_heads,
                gat_average_last
            )
        elif graph_layer_type == HeteroMAGNetBaseLayer.GraphLayerType.RGCN:
            self.relational_layer = RGCNModule(
                self.encoding_layer.out_features, relational_hidden_sizes, relational_output_size,
                num_node_types, agent_node_types, device, full_receptive_field, rgcn_n2_relations,
                rgcn_num_bases
            )

    def forward(self, data: Data):
        """"""
        # NOTE if the network receives a Batch of graphs, x, edge_index and node_type will still
        # be each a single Tensor, containing all graphs as a single disconnected graph.
        # So be careful when processing a batch not to confuse it with a single graph
        x, edge_index, node_type = data.x, data.edge_index, data.node_type

        x_by_class = {}
        for nt in node_type.unique().tolist():
            node_indices = (node_type == nt).nonzero().squeeze()  # grab nodes of that type
            in_size = self.node_type_input_sizes[nt]
            x_by_class[nt] = x[node_indices, :in_size
                               ]  # grab features only of those nodes, remove padding

        x = self.encoding_layer(x_by_class, node_type)
        x = self.relational_layer(x, edge_index, node_type)

        return x_by_class, x


class HeteroMAGNet(torch.nn.Module):

    def __init__(
        self,
        num_node_types: int,
        agent_node_types: list,
        node_type_input_sizes: list,
        encoding_hidden_sizes: list,
        encoding_output_size: int,
        relational_hidden_sizes: list,
        relational_output_size: int,
        device,
        graph_layer_type: HeteroMAGNetBaseLayer.GraphLayerType = HeteroMAGNetBaseLayer.
        GraphLayerType.GCN,
        full_receptive_field: bool = True,
        gat_n_heads: int = 1,
        gat_average_last: bool = False,
        rgcn_n2_relations: bool = True,
        rgcn_num_bases: int = 1,
        coalesce_states_and_obs: bool = False
    ):
        """Base class for the Heterogenous Multi-Agent Graph Network

        :param num_node_types: Number of node types
        :type num_node_types: int
        :param agent_node_types: List of which node types are agents
        :type agent_node_types: list
        :param node_type_input_sizes: Size of input vectors for each node type
        :type node_type_input_sizes: list
        :param encoding_hidden_sizes: List of number of neurons in the hidden layers of the encoding networks
        :type encoding_hidden_sizes: list
        :param encoding_output_size: Size of the output for the encoding networks
        :type encoding_output_size: int
        :param relational_hidden_sizes: List of number of neurons in the graph convolutional layers
        :type relational_hidden_sizes: int
        :param relational_output_size: Size of the output for the final graph convolutional layer
        :type relational_output_size: int
        :param device: PyTorch device
        :param graph_layer_type: which graph layer to use, defaults to GraphLayerType.GCN
        :type graph_layer_type: GraphLayerType, optional
        :param full_receptive_field: whether to return a concatenation of the outputs of all graph layers, defaults to True
        :type full_receptive_field: bool, optional
        :param gat_n_heads: number of heads for each GAT layer in a GAT module, defaults to 1
        :type gat_n_heads: int, optional
        :param gat_average_last: whether to average the heads of the last GAT, defaults to False (concatenation)
        :type gat_average_last: bool, optional
        :param rgcn_n2_relations: ``True``: generate one relation between each agent-node connection. ``False`` generate one relation for each agent node type, to be used against every other node, regardless of type, defaults to True
        :type rgcn_n2_relations: bool, optional
        :param rgcn_num_bases: number of basis matrices that compose each individual RGCN layer, defaults to 1
        :type rgcn_num_bases: int, optional
        :param coalesce_states_and_obs: whether to concatenate the original, unencoded feature vector of each agent with their final observation as input for the Q-networks, defaults to False
        :type coalesce_states_and_obs: bool, optional
        """
        super().__init__()

        self.device = device
        self.coalesce_states_and_obs = coalesce_states_and_obs
        self.agent_node_types = torch.tensor(agent_node_types, device=device)
        self.label_enc = LabelEncoder(agent_node_types)
        self.node_type_input_sizes = node_type_input_sizes

        self.base_layer = HeteroMAGNetBaseLayer(
            num_node_types, agent_node_types, node_type_input_sizes, encoding_hidden_sizes,
            encoding_output_size, relational_hidden_sizes, relational_output_size, device,
            graph_layer_type, full_receptive_field, gat_n_heads, gat_average_last,
            rgcn_n2_relations, rgcn_num_bases
        )

        self.action_layer: HeteroMAGNetActionLayer = None

    def _get_action_layer_input_sizes(self):
        # here we use relational_layer.out_features instead of relational_output_size
        # because the output size of some graph modules depend on more than the number of features,
        # like the GATModule
        act_layer_input_sizes = [self.base_layer.relational_layer.out_features
                                 ] * self.agent_node_types.size(0)
        if self.coalesce_states_and_obs:
            for i, agent_type in enumerate(self.agent_node_types):
                act_layer_input_sizes[i] += self.node_type_input_sizes[agent_type]

        return act_layer_input_sizes

    def forward(self, data):
        if self.action_layer is None:
            raise NotImplementedError(
                "This class does not directly implement the forward() method, please instantiate one of its base classes"
            )
        # NOTE if the network receives a Batch of graphs, x, edge_index and node_type will still
        # be each a single Tensor, containing all graphs as a single disconnected graph.
        # So be careful when processing a batch not to confuse it with a single graph
        x_by_class, x = self.base_layer(data)
        node_type = data.node_type
        agent_nodes = isin(node_type, self.agent_node_types)
        filtered_agent_node_types = node_type[agent_nodes]
        enc_node_types = torch.tensor(
            self.label_enc.transform(node_type[agent_nodes].cpu()),
            dtype=torch.long,
            device=self.device
        )

        obs_by_class = {}
        for nt in self.agent_node_types.tolist():
            agent_indices = (filtered_agent_node_types == nt
                             ).nonzero().squeeze()  # grab nodes of that type

            if not self.coalesce_states_and_obs:
                obs_by_class[nt] = x[agent_indices]
            else:
                obs_by_class[nt] = torch.cat((x_by_class[nt], x[agent_indices]), dim=-1)

        xdict = self.action_layer(obs_by_class, enc_node_types)

        return xdict


class HeteroMAGQNet(HeteroMAGNet):

    def __init__(
        self,
        num_node_types: int,
        agent_node_types: list,
        node_type_input_sizes: list,
        encoding_hidden_sizes: list,
        encoding_output_size: int,
        relational_hidden_sizes: list,
        relational_output_size: int,
        output_hidden_sizes: list,
        node_type_output_sizes: list,
        device,
        graph_layer_type: HeteroMAGNetBaseLayer.GraphLayerType = HeteroMAGNetBaseLayer.
        GraphLayerType.GCN,
        full_receptive_field: bool = True,
        gat_n_heads: int = 1,
        gat_average_last: bool = False,
        rgcn_n2_relations: bool = True,
        rgcn_num_bases: int = 1,
        coalesce_states_and_obs: bool = False
    ):
        """Heterogenous Multi-Agent Graph Q-Network, a variant of HMAQNet which uses one Q-network for each agent class to estimate Q-values for each class separately

        :param num_node_types: Number of node types
        :type num_node_types: int
        :param agent_node_types: List of which node types are agents
        :type agent_node_types: list
        :param node_type_input_sizes: Size of input vectors for each node type
        :type node_type_input_sizes: list
        :param encoding_hidden_sizes: List of number of neurons in the hidden layers of the encoding networks
        :type encoding_hidden_sizes: list
        :param encoding_output_size: Size of the output for the encoding networks
        :type encoding_output_size: int
        :param relational_hidden_sizes: List of number of neurons in the graph convolutional layers
        :type relational_hidden_sizes: int
        :param relational_output_size: Size of the output for the final graph convolutional layer
        :type relational_output_size: int
        :param output_hidden_sizes: List of number of neurons in the hidden layers of the Q-networks
        :type output_hidden_sizes: list
        :param node_type_output_sizes: Number of actions for each agent node type
        :type node_type_output_sizes: list
        :param device: PyTorch device
        :param graph_layer_type: which graph layer to use, defaults to HeteroMAGNetBaseLayer.GraphLayerType.GCN
        :type graph_layer_type: HeteroMAGNetBaseLayer.GraphLayerType, optional
        :param full_receptive_field: whether to return a concatenation of the outputs of all graph layers, defaults to True
        :type full_receptive_field: bool, optional
        :param gat_n_heads: number of heads for each GAT layer in a GAT module, defaults to 1
        :type gat_n_heads: int, optional
        :param gat_average_last: whether to average the heads of the last GAT, defaults to False (concatenation)
        :type gat_average_last: bool, optional
        :param rgcn_n2_relations: ``True``: generate one relation between each agent-node connection. ``False`` generate one relation for each agent node type, to be used against every other node, regardless of type, defaults to True
        :type rgcn_n2_relations: bool, optional
        :param rgcn_num_bases: number of basis matrices that compose each individual RGCN layer, defaults to 1
        :type rgcn_num_bases: int, optional
        :param coalesce_states_and_obs: whether to concatenate the original, unencoded feature vector of each agent with their final observation as input for the Q-networks, defaults to False
        :type coalesce_states_and_obs: bool, optional
        """
        super().__init__(
            num_node_types, agent_node_types, node_type_input_sizes, encoding_hidden_sizes,
            encoding_output_size, relational_hidden_sizes, relational_output_size, device,
            graph_layer_type, full_receptive_field, gat_n_heads, gat_average_last,
            rgcn_n2_relations, rgcn_num_bases, coalesce_states_and_obs
        )

        act_layer_input_sizes = self._get_action_layer_input_sizes()

        self.action_layer = MultiAgentMultiNetQLayer(
            act_layer_input_sizes, output_hidden_sizes,
            [node_type_output_sizes[u] for u in agent_node_types], device
        )


class HeteroMAGNetA2CMLP(HeteroMAGNet):

    def __init__(
        self,
        num_node_types: int,
        agent_node_types: list,
        node_type_input_sizes: list,
        encoding_hidden_sizes: list,
        encoding_output_size: int,
        relational_hidden_sizes: list,
        relational_output_size: int,
        output_hidden_sizes: list,
        node_type_output_sizes: list,
        device,
        graph_layer_type: HeteroMAGNetBaseLayer.GraphLayerType = HeteroMAGNetBaseLayer.
        GraphLayerType.GCN,
        full_receptive_field: bool = True,
        gat_n_heads: int = 1,
        gat_average_last: bool = False,
        rgcn_n2_relations: bool = True,
        rgcn_num_bases: int = 1,
        coalesce_states_and_obs: bool = False
    ):
        """Heterogenous Multi-Agent Actor-Critic Graph Network, a variant of HMAGNet which uses one policy head and one value head for each agent class to estimate action probabilities and state values for each class separately.

        - `A2C blog post <https://openai.com/blog/baselines-acktr-a2c/#a2canda3c>`_

        - `A2C code example <https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb>`_

        - `A3C code example <https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2>`_

        :param num_node_types: Number of node types
        :type num_node_types: int
        :param agent_node_types: List of which node types are agents
        :type agent_node_types: list
        :param node_type_input_sizes: Size of input vectors for each node type
        :type node_type_input_sizes: list
        :param encoding_hidden_sizes: List of number of neurons in the hidden layers of the encoding networks
        :type encoding_hidden_sizes: list
        :param encoding_output_size: Size of the output for the encoding networks
        :type encoding_output_size: int
        :param relational_hidden_sizes: List of number of neurons in the graph convolutional layers
        :type relational_hidden_sizes: int
        :param relational_output_size: Size of the output for the final graph convolutional layer
        :type relational_output_size: int
        :param output_hidden_sizes: List of number of neurons in the hidden layers of the Q-networks
        :type output_hidden_sizes: list
        :param node_type_output_sizes: Number of actions for each agent node type
        :type node_type_output_sizes: list
        :param device: PyTorch device
        :param graph_layer_type: which graph layer to use, defaults to HeteroMAGNetBaseLayer.GraphLayerType.GCN
        :type graph_layer_type: HeteroMAGNetBaseLayer.GraphLayerType, optional
        :param full_receptive_field: whether to return a concatenation of the outputs of all graph layers, defaults to True
        :type full_receptive_field: bool, optional
        :param gat_n_heads: number of heads for each GAT layer in a GAT module, defaults to 1
        :type gat_n_heads: int, optional
        :param gat_average_last: whether to average the heads of the last GAT, defaults to False (concatenation)
        :type gat_average_last: bool, optional
        :param rgcn_n2_relations: ``True``: generate one relation between each agent-node connection. ``False`` generate one relation for each agent node type, to be used against every other node, regardless of type, defaults to True
        :type rgcn_n2_relations: bool, optional
        :param rgcn_num_bases: number of basis matrices that compose each individual RGCN layer, defaults to 1
        :type rgcn_num_bases: int, optional
        :param coalesce_states_and_obs: whether to concatenate the original, unencoded feature vector of each agent with their final observation as input for the Q-networks, defaults to False
        :type coalesce_states_and_obs: bool, optional
        """
        super().__init__(
            num_node_types, agent_node_types, node_type_input_sizes, encoding_hidden_sizes,
            encoding_output_size, relational_hidden_sizes, relational_output_size, device,
            graph_layer_type, full_receptive_field, gat_n_heads, gat_average_last,
            rgcn_n2_relations, rgcn_num_bases, coalesce_states_and_obs
        )

        act_layer_input_sizes = self._get_action_layer_input_sizes()

        self.action_layer = MultiAgentMultiNetActorCriticLayer(
            act_layer_input_sizes, output_hidden_sizes,
            [node_type_output_sizes[u] for u in agent_node_types], device
        )

    # loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
    # return values, xdict


class HeteroMAGNetPPO(HeteroMAGNet):

    def __init__(
        self,
        num_node_types: int,
        agent_node_types: list,
        node_type_input_sizes: list,
        encoding_hidden_sizes: list,
        encoding_output_size: int,
        relational_hidden_sizes: list,
        relational_output_size: int,
        output_hidden_sizes: list,
        node_type_output_sizes: list,
        device,
        graph_layer_type: HeteroMAGNetBaseLayer.GraphLayerType = HeteroMAGNetBaseLayer.
        GraphLayerType.GCN,
        full_receptive_field: bool = True,
        gat_n_heads: int = 1,
        gat_average_last: bool = False,
        rgcn_n2_relations: bool = True,
        rgcn_num_bases: int = 1,
        coalesce_states_and_obs: bool = False
    ):
        """- `PPO paper <https://arxiv.org/abs/1707.06347>`_

        - `pseudocode <https://arxiv.org/abs/1707.02286>`_

        - `human readable code <https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/12_Proximal_Policy_Optimization>`_

        - I know the author of `this one <https://github.com/vwxyzjn/gym-microrts/blob/master/experiments/ppo2.py>`_

        :param num_node_types: Number of node types
        :type num_node_types: int
        :param agent_node_types: List of which node types are agents
        :type agent_node_types: list
        :param node_type_input_sizes: Size of input vectors for each node type
        :type node_type_input_sizes: list
        :param encoding_hidden_sizes: List of number of neurons in the hidden layers of the encoding networks
        :type encoding_hidden_sizes: list
        :param encoding_output_size: Size of the output for the encoding networks
        :type encoding_output_size: int
        :param relational_hidden_sizes: List of number of neurons in the graph convolutional layers
        :type relational_hidden_sizes: int
        :param relational_output_size: Size of the output for the final graph convolutional layer
        :type relational_output_size: int
        :param output_hidden_sizes: List of number of neurons in the hidden layers of the Q-networks
        :type output_hidden_sizes: list
        :param node_type_output_sizes: Number of actions for each agent node type
        :type node_type_output_sizes: list
        :param device: PyTorch device
        :param graph_layer_type: which graph layer to use, defaults to HeteroMAGNetBaseLayer.GraphLayerType.GCN
        :type graph_layer_type: HeteroMAGNetBaseLayer.GraphLayerType, optional
        :param full_receptive_field: whether to return a concatenation of the outputs of all graph layers, defaults to True
        :type full_receptive_field: bool, optional
        :param gat_n_heads: number of heads for each GAT layer in a GAT module, defaults to 1
        :type gat_n_heads: int, optional
        :param gat_average_last: whether to average the heads of the last GAT, defaults to False (concatenation)
        :type gat_average_last: bool, optional
        :param rgcn_n2_relations: ``True``: generate one relation between each agent-node connection. ``False`` generate one relation for each agent node type, to be used against every other node, regardless of type, defaults to True
        :type rgcn_n2_relations: bool, optional
        :param rgcn_num_bases: number of basis matrices that compose each individual RGCN layer, defaults to 1
        :type rgcn_num_bases: int, optional
        :param coalesce_states_and_obs: whether to concatenate the original, unencoded feature vector of each agent with their final observation as input for the Q-networks, defaults to False
        :type coalesce_states_and_obs: bool, optional
        """
        super().__init__(
            num_node_types, agent_node_types, node_type_input_sizes, encoding_hidden_sizes,
            encoding_output_size, relational_hidden_sizes, relational_output_size, device,
            graph_layer_type, full_receptive_field, gat_n_heads, gat_average_last,
            rgcn_n2_relations, rgcn_num_bases, coalesce_states_and_obs
        )
