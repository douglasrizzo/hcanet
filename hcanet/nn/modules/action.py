from enum import Enum

import torch as th
import torch.nn.functional as F
from torch import nn
from torch.cuda import device as torch_device

from ..activation import get_activation


class HeteroMAGNetActionLayer(nn.Module):

   class LayerType(Enum):
      DQN = 'Deep Q-Network'
      DDQN = 'Dueling Deep Q-Network'

   def __init__(self,
                n_agents_by_net: list,
                input_size: int,
                hidden_size: int,
                output_sizes: list,
                use_rnn: bool,
                activation: str,
                device: torch_device):
      super().__init__()

      self.n_agents_by_net = n_agents_by_net
      self.device = device
      self.use_rnn = use_rnn
      self.output_sizes = output_sizes
      self.layer1 = nn.ModuleList()
      self.hidden_states: list = []
      self.activation = get_activation(activation)

      for _ in range(len(n_agents_by_net)):
         if self.use_rnn:
            self.layer1.append(th.nn.LSTMCell(input_size, hidden_size))
         else:
            self.layer1.append(th.nn.Linear(input_size, hidden_size))

      self.max_output_dim = None
      if len(set(self.output_sizes)) > 1:
         self.max_output_dim = max(self.output_sizes)

   def forward(self, x: dict, node_type) -> th.tensor:
      """"""
      raise NotImplementedError(
          "This class does not directly implement the forward() method, please instantiate one of its base classes"
      )

   def init_hidden(self, batch_size: int):
      assert isinstance(self.layer1[0], th.nn.LSTMCell)
      #  and len(n_agents) == len(self.layer1) == self.expected_n_agents()

      self.hidden_states = []
      for i in range(len(self.layer1)):
         hs = th.zeros(self.n_agents_by_net[i] * batch_size,
                       self.layer1[i].hidden_size,
                       device=self.device)
         cs = th.zeros(self.n_agents_by_net[i] * batch_size,
                       self.layer1[i].hidden_size,
                       device=self.device)
         self.hidden_states.append((hs, cs))

   def apply_net(self, index, x: th.tensor):
      # remove singleton dim
      x.squeeze_()
      if x.ndim == 1:
         x = x.unsqueeze(0)
      assert x.ndim == 2, "only agent dim and feature dim here!"

      l1 = self.layer1[index]

      if self.use_rnn:
         hidden, cell = self.hidden_states[index]
         x, cell = l1(x, (hidden, cell))
         self.hidden_states[index] = (x, cell)
      else:
         x = l1(x)

      return self.activation(x)

   def _pad(self, output_dict: dict):
      agent_indices = th.cat([output_dict[nt][0] for nt in output_dict])

      agent_qs = [output_dict[nt][1] for nt in output_dict]

      if self.max_output_dim is not None:
         agent_qs = [F.pad(agent_q, agent_q, 0, 0) for agent_q in agent_qs]

      agent_qs = th.cat(agent_qs)[agent_indices]

      agent_vs = None
      if len(output_dict[0]) == 3:
         agent_vs = th.cat([output_dict[nt][2] for nt in output_dict])[agent_indices]

      if agent_vs is None:
         return agent_qs
      else:
         return agent_qs, agent_vs


class QLayer(HeteroMAGNetActionLayer):
   """An action layer which approximates one vector for each agent/agent class/all agents. Can be used to approximate Q-values.

   :param HeteroMAGNetActionLayer: [description]
   :type HeteroMAGNetActionLayer: [type]
   """

   def __init__(self,
                n_agents_by_net,
                input_size: int,
                hidden_size: int,
                output_sizes: list,
                use_rnn: bool,
                activation: str,
                device: torch_device):
      super().__init__(n_agents_by_net,
                       input_size,
                       hidden_size,
                       output_sizes,
                       use_rnn,
                       activation,
                       device)
      self.layer2 = nn.ModuleList()
      for i in range(len(output_sizes)):
         out_size = output_sizes[i]

         #   h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
         self.layer2.append(nn.Linear(hidden_size, out_size))

      def init_xavier(m):
         if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('sigmoid'))

      for nets in (self.layer1, self.layer2):
         for net in nets:
            if net is not None:
               net.apply(init_xavier)

   def forward(self, x: dict, node_type) -> dict:
      output_dict = {}

      # for each agent node type, retrieve their q-network and
      # calculate q(s) for all agents of the same type, as a batch
      # if there are multiple
      unique_types = node_type.unique().tolist()

      for nt in unique_types:
         # get indices of nodes that belong to the current class
         node_indices = (node_type == nt).nonzero().squeeze(1)
         output = self.layer2[nt](self.apply_net(nt, x[nt]))
         output_dict[nt] = node_indices, output
      del node_indices, output

      # fix dimensions for networks that processed a single agent
      for nt in unique_types:
         if output_dict[nt][1].ndim == 1:
            output_dict[nt] = output_dict[nt][0], output_dict[nt][1].unsqueeze(0)

      output_dict = self._pad(output_dict)

      return output_dict


class ActorCriticLayer(HeteroMAGNetActionLayer):

   def __init__(self,
                node_types,
                input_size: int,
                hidden_size: int,
                output_sizes: list,
                use_rnn: bool,
                activation: str,
                device: torch_device,
                dueling_dqn=False):
      """An action layer which uses two individual output models for each agent/agent class/all agents to approximate one vector and one scalar. Can be used to approximate policy distributions/action-values and state-values/advantages.
      """
      super().__init__(node_types,
                       input_size,
                       hidden_size,
                       output_sizes,
                       use_rnn,
                       activation,
                       device)

      self.policy_heads = nn.ModuleList()
      self.value_heads = nn.ModuleList()
      self.dueling_dqn = dueling_dqn

      for i in range(len(output_sizes)):
         out_size = output_sizes[i]

         #   h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

         self.policy_heads.append(
             nn.Sequential(nn.Linear(hidden_size, out_size)) if self.dueling_dqn else nn.
             Sequential(nn.Linear(hidden_size, out_size), th.nn.Softmax()))

         self.value_heads.append(nn.Linear(hidden_size, 1))

      def init_xavier(m):
         if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('sigmoid'))

      for nets in (self.layer1, self.policy_heads, self.value_heads):
         for net in nets:
            if net is not None:
               net.apply(init_xavier)

   def forward(self, x: dict, node_type):
      output_dict = {}

      # for each agent node type, retrieve their q-network and
      # calculate q(s) for all agents of the same type, as a batch
      # if there are multiple
      unique_types = node_type.unique().tolist()

      for nt in unique_types:
         node_indices = (node_type == nt).nonzero().squeeze(1)
         policy_head = self.policy_heads[nt]
         value_head = self.value_heads[nt]

         # get the correct networks/heads
         # apply it to the data, keep in a dict alongside node indices
         intermediate_value = self.apply_net(nt, x[nt])
         # NOTE it's important that the first two arguments are passed in this order, like the Q-network does
         output_dict[nt] = node_indices, policy_head(intermediate_value), value_head(
               intermediate_value).squeeze()
      del node_indices, intermediate_value

      for nt in unique_types:
         # fix dimensions for networks that processed a single agent
         if output_dict[nt][1].ndim == 1:
            output_dict[nt][1].unsqueeze_(0)
            if len(output_dict[nt]) == 3:
               output_dict[nt][2].unsqueeze_(0)
         if self.dueling_dqn:
            if output_dict[nt][2].ndim == 0:
               output_dict[nt][2].unsqueeze_(0)

            output_dict[nt] = output_dict[nt][0], output_dict[nt][2].unsqueeze(1) + output_dict[nt][1] - output_dict[nt][1].mean()

      output = self._pad(output_dict)

      return output
