import torch as th
from torch_geometric.nn import FastRGCNConv, GATConv, GCNConv, RGCNConv

from ...encoding import isin
from ..activation import get_activation


class GraphModule(th.nn.Module):
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
      num_node_types: int,
      agent_node_types: list,
      activation: str,
      device,
      full_receptive_field: bool = True,
   ):
      super().__init__()
      self.device = device
      self.num_node_types = num_node_types
      self.agent_node_types = th.tensor(agent_node_types, device=self.device)
      self.layers = th.nn.ModuleList()
      self.full_receptive_field = full_receptive_field
      self.activation = get_activation(activation)

      if hidden_sizes is None:
         hidden_sizes = []

      self._sizes = [input_size] + hidden_sizes

      if full_receptive_field:
         self.out_features = sum(hidden_sizes)
      else:
         self.out_features = self._sizes[-1]

   def _get_agent_nodes(self, node_type):
      return isin(node_type, self.agent_node_types)


class GCNModule(GraphModule):
   """Communication module composed of plain graph convolutional layers"""
   def __init__(
      self,
      input_size: int,
      hidden_sizes: list,
      num_node_types: int,
      agent_node_types: list,
      activation: str,
      device,
      full_receptive_field: bool = True,
   ):
      super().__init__(
         input_size,
         hidden_sizes,
         num_node_types,
         agent_node_types,
         activation,
         device,
         full_receptive_field,
      )

      for i in range(len(self._sizes) - 1):
         self.layers.append(GCNConv(self._sizes[i], self._sizes[i + 1]))

   def forward(self, x, edge_index, node_type):
      agent_nodes = self._get_agent_nodes(node_type)

      final_x = []

      for i in range(len(self.layers)):
         x = self.layers[i](x, edge_index)
         x = self.activation(x)
         # x = F.dropout(x, training=self.training)

         if self.full_receptive_field or i == len(self.layers) - 1:
            final_x.append(x[agent_nodes])

      return th.cat(final_x, dim=1)


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
      num_node_types: int,
      agent_node_types: list,
      activation: str,
      device,
      full_receptive_field: bool = True,
      n_heads: int = 2,
      average_last: bool = False,
   ):
      super().__init__(
         input_size,
         hidden_sizes,
         num_node_types,
         agent_node_types,
         activation,
         device,
         full_receptive_field,
      )

      self.last_layer_attention = None
      self.attention_indices = None
      self.n_heads = n_heads

      self._sizes = [input_size] + hidden_sizes

      self.out_features = (hidden_sizes[-1] if average_last else hidden_sizes[-1] * n_heads)
      if self.full_receptive_field:
         self.out_features += sum(h * n_heads for h in hidden_sizes[:-1])

      # build each layer
      for i in range(len(self._sizes) - 1):
         # input and output size of each layer
         # if the previous layer has multiple heads, its output is n_heads times larger,
         # so we account for that here
         # in_size = self._sizes[i] if i == 0 else self._sizes[i] * n_heads
         # out_size = self._sizes[i + 1]
         in_size = self._sizes[i] if i == 0 else self._sizes[i] * n_heads
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
            x, (self.attention_indices, self.last_layer_attention) = self.layers[i](
               x, edge_index, return_attention_weights=True)

         x = self.activation(x)

         if self.full_receptive_field or i == len(self.layers) - 1:
            final_x.append(x[agent_nodes])

      return th.cat(final_x, dim=1)


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
      num_node_types: int,
      agent_node_types: list,
      activation: str,
      device,
      full_receptive_field: bool = True,
      n2_edge_types: bool = True,
      num_bases: int = 1,
      fast: bool = False,
   ):
      super().__init__(
         input_size,
         hidden_sizes,
         num_node_types,
         agent_node_types,
         activation,
         device,
         full_receptive_field,
      )

      self.n2_edge_types = n2_edge_types

      # if we have N node types, we want the RGCNConv layer to be able to model
      # one type of relation for each possible pair of node type
      self.num_possible_relations = (len(agent_node_types) *
                                     num_node_types if n2_edge_types else num_node_types)

      conv = RGCNConv if not fast else FastRGCNConv
      for i in range(len(self._sizes) - 1):
         self.layers.append(
            conv(
               self._sizes[i],
               self._sizes[i + 1],
               self.num_possible_relations,
               num_bases,
            ))

   def _generate_edge_types(self, edge_index: th.tensor, node_type: th.tensor) -> th.tensor:
      if self.n2_edge_types:
         edge_type = (node_type[edge_index[1]] * self.num_node_types + node_type[edge_index[0]])
      else:
         edge_type = node_type[edge_index[1]]

      return edge_type

   def forward(self, x: th.tensor, edge_index: th.tensor, node_type: th.tensor) -> th.tensor:
      edge_type = self._generate_edge_types(edge_index, node_type)
      agent_nodes = self._get_agent_nodes(node_type)

      final_x = []

      for i in range(len(self.layers)):
         x = self.layers[i](x, edge_index, edge_type)
         x = self.activation(x)
         # x = F.dropout(x, training=self.training)

         if self.full_receptive_field or i == len(self.layers) - 1:
            final_x.append(x[agent_nodes])

      return th.cat(final_x, dim=1)
