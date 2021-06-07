from enum import Enum

import torch as th
from torch_geometric.data import Batch

from ..encoding import isin
from .modules.action import ActorCriticLayer, HeteroMAGNetActionLayer, QLayer
from .modules.encoding import EncoderByType
from .modules.graph import GATModule, GCNModule, GraphModule, RGCNModule
from .modules.mixing import VDNMixer


class HeteroMAGNet(th.nn.Module):

   class GraphLayerType(Enum):
      GCN = 'GCN'
      GAT = 'GAT'
      RGCN = 'RGCN'

   def __init__(self,
                action_layer_type: HeteroMAGNetActionLayer.LayerType,
                share_encoding: bool,
                share_comms: bool,
                share_action: bool,
                num_node_types: int,
                agent_node_types: list,
                node_types: list,
                features_by_node_class: list,
                encoding_output_size: int,
                graph_module_sizes: list,
                action_hidden_size: int,
                action_output_sizes: list,
                act_encoding: str,
                act_comms: str,
                act_action: str,
                use_rnn_encoding: bool,
                use_rnn_action: bool,
                device,
                graph_layer_type: GraphLayerType = None,
                full_receptive_field: bool = True,
                gat_n_heads: int = 1,
                gat_average_last: bool = False,
                rgcn_n2_relations: bool = True,
                rgcn_num_bases: int = 1,
                rgcn_fast: bool = False,
                mixing=None):
      super().__init__()

      self.device = device
      self.graph_layer_type = graph_layer_type

      self.action_layer: HeteroMAGNetActionLayer

      # NOTE this assumes all agents have the same number of actions
      self.n_actions = action_output_sizes[0]

      n_agents_by_net = [node_types.count(ai) for ai in sorted(set(node_types))]
      individual_node_types = th.tensor(node_types, device=device)
      shared_node_types = th.zeros(len(node_types), device=device, dtype=th.int)

      if share_encoding:
         n_agents_by_net_encoding = [sum(n_agents_by_net)]
         self.node_types_encoding = shared_node_types

         assert len(set(features_by_node_class)) == 1, 'Number of inputs by node class must all be equal when sharing parameters from the encoding layer'
         self.features_by_node_class = list(set(features_by_node_class))
      else:
         n_agents_by_net_encoding = n_agents_by_net
         self.node_types_encoding = individual_node_types

         assert len(features_by_node_class) == len(n_agents_by_net_encoding), 'Number of node classes and number of inputs must be the same when not sharing parameters from the encoding layer'
         self.features_by_node_class = features_by_node_class

      self.node_types_comms = individual_node_types if not share_comms else shared_node_types

      if share_action:
         n_agents_by_net_action = [sum(n_agents_by_net)]
         n_actions_by_agent_class = [self.n_actions]
         self.node_types_action = shared_node_types
      else:
         n_agents_by_net_action = n_agents_by_net
         n_actions_by_agent_class = [action_output_sizes[u] for u in agent_node_types]
         self.node_types_action = individual_node_types

      self.encoding_layer = EncoderByType(n_agents_by_net_encoding,
                                          self.features_by_node_class,
                                          encoding_output_size,
                                          use_rnn_encoding,
                                          act_encoding,
                                          device)

      self.relational_layer = None
      if graph_layer_type == self.GraphLayerType.GCN:
         self.relational_layer = GCNModule(self.encoding_layer.out_features,
                                           graph_module_sizes,
                                           num_node_types,
                                           agent_node_types,
                                           act_comms,
                                           device,
                                           full_receptive_field)  # type: GraphModule
      elif graph_layer_type == self.GraphLayerType.GAT:
         self.relational_layer = GATModule(self.encoding_layer.out_features,
                                           graph_module_sizes,
                                           num_node_types,
                                           agent_node_types,
                                           act_comms,
                                           device,
                                           full_receptive_field,
                                           gat_n_heads,
                                           gat_average_last)
      elif graph_layer_type == self.GraphLayerType.RGCN:
         self.relational_layer = RGCNModule(self.encoding_layer.out_features,
                                            graph_module_sizes,
                                            num_node_types,
                                            agent_node_types,
                                            act_comms,
                                            device,
                                            full_receptive_field,
                                            rgcn_n2_relations,
                                            rgcn_num_bases,
                                            rgcn_fast)

      # here we use relational_layer.out_features instead of relational_output_size
      # because the output size of some graph modules depend on more than the number of features,
      # like the GATModule
      act_layer_input_size = self.relational_layer.out_features if self.relational_layer is not None else self.encoding_layer.out_features

      if action_layer_type == HeteroMAGNetActionLayer.LayerType.DQN:
         self.action_layer = QLayer(n_agents_by_net_action,
                                    act_layer_input_size,
                                    action_hidden_size,
                                    n_actions_by_agent_class,
                                    use_rnn_action,
                                    act_action,
                                    device)

      elif action_layer_type == HeteroMAGNetActionLayer.LayerType.DDQN:
         self.action_layer = ActorCriticLayer(
             n_agents_by_net_action,
             act_layer_input_size,
             action_hidden_size,
             n_actions_by_agent_class,
             use_rnn_action,
             act_action,
             device,
             action_layer_type == HeteroMAGNetActionLayer.LayerType.DDQN)

      self.mixer = None
      if mixing == 'vdn':
         self.mixer = VDNMixer()

   def forward(self, data):
      """"""
      # NOTE if data is a Batch object, x, edge_index and node_type will still
      # be low-dimensional Tensors, containing all graphs in a single disconnected graph.
      # So be careful when processing a batch not to confuse it with a single graph

      # get data
      x, edge_index = data.x, data.edge_index

      # get batch size (single state = batch of size 1)
      bs = 1
      if isinstance(data, Batch) and data.num_graphs > 1:
         bs = data.num_graphs

      batch_node_types_encoding = self.node_types_encoding.repeat(bs)
      batch_node_types_comms = self.node_types_comms.repeat(bs)
      batch_node_types_action = self.node_types_action.repeat(bs)

      # separate input tensor into tensors pertaining to individual agent classes
      # according to what the encoding module expects
      input_by_class = {}
      for nt in self.node_types_encoding.unique():
         # grab nodes of the current class
         node_mask = (batch_node_types_encoding == nt)
         # grab features only of those nodes, remove padding
         in_size = self.features_by_node_class[int(nt)]
         input_by_class[int(nt)] = x[node_mask, :in_size]
      del node_mask

      # apply encoding layer, output is a single tensor of size n_agents x encoding_size
      x = self.encoding_layer(input_by_class, batch_node_types_encoding)
      del input_by_class

      # if the communication layer exists, apply it to the data
      # output is also a single tensor of size n_agents x comms_output_size
      if self.relational_layer is not None:
         x = self.relational_layer(x, edge_index, batch_node_types_comms)

      # NOTE this is where I used to filter agent nodes from non-agent nodes,
      # as well as their features, but, as of the time of this writing,
      # all nodes are agent nodes

      obs_by_class = {}
      for nt in self.node_types_action.unique():
         # grab nodes of that type
         agent_mask = (batch_node_types_action == nt)
         obs_by_class[int(nt)] = x[agent_mask]
         # obs_by_class[nt] = th.cat((input_by_class[nt], x[agent_indices]), dim=-1)

         # input for recurrent layers is
         # (number of sequences) (sequence size) (features of individual elements)
         # translated for this network, it should be
         # (episodes) (steps in episode) (individual steps)

         # episodes = my batch size, bs
         # steps in episode = total number of nodes / batch_size
         # batch_size = number of graphs in the Batch object, or 1 if Data
         # size of previous layer = self.relational_layer.out_features

         # by my interpretation, this should be the correct size
         # obs_by_class[nt] = obs_by_class[nt].view(
         #     agent_indices.size(0) // bs, bs, self.relational_layer.out_features)

         # however, it looks like it should be like this?!
         # obs_by_class[nt] = obs_by_class[nt].view(
         #  agent_indices.size(0) // bs, bs, self.relational_layer.out_features)
      x = self.action_layer(obs_by_class, batch_node_types_action)

      return x.view(bs, -1, self.n_actions)
