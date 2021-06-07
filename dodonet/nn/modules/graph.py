import torch
from torch_geometric.nn import GATConv, GCNConv, RGCNConv

from dodonet.encoding import isin


class GraphModule(torch.nn.Module):
    """Base class for graph communication modules

    :param input_size: number of input features for all nodes
    :type input_size: int
    :param hidden_sizes: list of the sizes of hidden layers
    :type hidden_sizes: list
    :param out_size: number of output features
    :type out_size: int
    :param num_node_types: total number of possible node types
    :type num_node_types: int
    :param agent_node_types: which node types from 0 to ``num_node_types``, represent agent nodes
    :type agent_node_types: list
    :param device: PyTorch device to store tensors
    :param full_receptive_field: return a concatenation of the outputs of all layers from this module, instead of the output of only the last layer. Defaults to True
    :type full_receptive_field: bool, optional
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        out_size: int,
        num_node_types: int,
        agent_node_types: list,
        device,
        full_receptive_field: bool = True
    ):
        super().__init__()
        self.device = device
        self.num_node_types = num_node_types
        self.agent_node_types = torch.tensor(agent_node_types, device=self.device)
        self.layers = torch.nn.ModuleList()
        self.full_receptive_field = full_receptive_field

        if hidden_sizes is None:
            hidden_sizes = []

        self.in_features = input_size
        self.out_features = out_size

        if full_receptive_field:
            self.out_features += sum(hidden_sizes)

        self._sizes = [input_size] + hidden_sizes + [out_size]

    def _get_agent_nodes(self, node_type):
        return isin(node_type, self.agent_node_types)


class GCNModule(GraphModule):
    """Communication module composed of plain graph convolutional layers"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        out_size: int,
        num_node_types: int,
        agent_node_types: list,
        device,
        full_receptive_field: bool = True,
    ):
        super().__init__(
            input_size, hidden_sizes, out_size, num_node_types, agent_node_types, device,
            full_receptive_field
        )

        for i in range(len(self._sizes) - 1):
            self.layers.append(GCNConv(self._sizes[i], self._sizes[i + 1]))

    def forward(self, x, edge_index, node_type):
        agent_nodes = self._get_agent_nodes(node_type)

        final_x = []

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = torch.sigmoid(x)
            # x = F.dropout(x, training=self.training)

            if self.full_receptive_field or i == len(self.layers) - 1:
                final_x.append(x[agent_nodes])

        return torch.cat(final_x, dim=1)


class GATModule(GraphModule):
    """Communication module composed of GAT layers

    :param n_heads: number of attention heads, defaults to 2
    :type n_heads: int, optional
    :param average_last: average the last layer, else concatenate, defaults to False
    :type average_last: bool, optional
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        out_size: int,
        num_node_types: int,
        agent_node_types: list,
        device,
        full_receptive_field: bool = True,
        n_heads: int = 2,
        average_last: bool = False
    ):
        super().__init__(
            input_size, hidden_sizes, out_size, num_node_types, agent_node_types, device,
            full_receptive_field
        )

        self.last_layer_attention = None
        self.attention_indices = None
        self.out_features = self._sizes[-1] if average_last else self._sizes[-1] * n_heads
        self.n_heads = n_heads

        if self.full_receptive_field:
            self.out_features += sum(h * n_heads for h in hidden_sizes)

        self._sizes = [input_size] + hidden_sizes + [out_size]

        for i in range(len(self._sizes) - 1):
            in_size = self._sizes[i] if i == 0 else out_size * n_heads

            out_size = self._sizes[i + 1]

            concat = i < len(self._sizes) - 2 or not average_last
            self.layers.append(GATConv(in_size, out_size, n_heads, concat=concat))

    def forward(self, x, edge_index, node_type):
        agent_nodes = self._get_agent_nodes(node_type)

        final_x = []

        for i in range(len(self.layers)):
            if i != len(self.layers) - 1:
                x = self.layers[i](x, edge_index)
            else:
                x, (self.attention_indices, self.last_layer_attention
                    ) = self.layers[i](x, edge_index, return_attention_weights=True)

            x = torch.sigmoid(x)

            if self.full_receptive_field or i == len(self.layers) - 1:
                final_x.append(x[agent_nodes])

        return torch.cat(final_x, dim=1)


class RGCNModule(GraphModule):
    """Communication module composed of relational graph convolutional layers

    :param n2_edge_types: ``True``: generate one relation between each agent-node connection. ``False`` generate one relation for each agent node type, to be used against every other node, regardless of type, defaults to True
    :type n2_edge_types: bool, optional
    :param num_bases: number of basis matrices that compose each individual RGCN layer, defaults to 1
    :type num_bases: int, optional
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        out_size: int,
        num_node_types: int,
        agent_node_types: list,
        device,
        full_receptive_field: bool = True,
        n2_edge_types: bool = True,
        num_bases: int = 1
    ):
        super().__init__(
            input_size, hidden_sizes, out_size, num_node_types, agent_node_types, device,
            full_receptive_field
        )

        self.n2_edge_types = n2_edge_types

        # if we have N node types, we want the RGCNConv layer to be able to model
        # one type of relation for each possible pair of node type
        self.num_possible_relations = len(
            agent_node_types
        ) * num_node_types if n2_edge_types else num_node_types

        # we have to make sure that, inside this class, agent classes are numbered starting from 0
        # and the other node classes start from len(agent_node_type)
        # this way, when we want the relation index between an agent and any other node,
        # we can calculate is normalized_agent_class * max_num_relations + normalized_other_node_class
        self.ordered_node_indices = {}
        agent_value = 0
        nonagent_value = len(agent_node_types)
        for node_type in range(num_node_types):
            if node_type in agent_node_types:
                self.ordered_node_indices[node_type] = agent_value
                agent_value += 1
            elif n2_edge_types:  # we only keep normalized non-agent classes if we need to calculate n2 relations
                self.ordered_node_indices[node_type] = nonagent_value
                nonagent_value += 1

        for i in range(len(self._sizes) - 1):
            self.layers.append(
                RGCNConv(
                    self._sizes[i], self._sizes[i + 1], self.num_possible_relations, num_bases
                )
            )

    def _get_relation(self, source_node_type: int, target_node_type: int) -> int:
        if target_node_type not in self.agent_node_types:
            raise ValueError("Agent node is not target of the edge")

        relation = self.ordered_node_indices[target_node_type]

        if self.n2_edge_types:
            normalized_source_type = self.ordered_node_indices[source_node_type]
            relation = relation * self.num_node_types + normalized_source_type

        assert relation < self.num_possible_relations
        return relation

    def _generate_edge_types(
        self, edge_index: torch.Tensor, node_type: torch.Tensor
    ) -> torch.Tensor:
        edge_type = torch.tensor(
            [
                self._get_relation(int(node_type[source_node]), int(node_type[target_node]))
                for (source_node, target_node) in edge_index.T
            ],
            dtype=torch.long,
            device=self.device
        )

        return edge_type

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor
    ) -> torch.Tensor:
        edge_type = self._generate_edge_types(edge_index, node_type)
        agent_nodes = self._get_agent_nodes(node_type)

        final_x = []

        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index, edge_type)
            x = torch.sigmoid(x)
            # x = F.dropout(x, training=self.training)

            if self.full_receptive_field or i == len(self.layers) - 1:
                final_x.append(x[agent_nodes])

        return torch.cat(final_x, dim=1)
