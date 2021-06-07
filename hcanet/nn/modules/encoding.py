import torch as th
from torch.cuda import device as torch_device

from ..activation import get_activation


class EncoderByType(th.nn.Module):

   def __init__(self,
                n_agents_by_net: list,
                n_inputs_by_node_class: list,
                encoding_size: int,
                use_rnn: bool,
                activation: str,
                device: torch_device):
      super().__init__()
      self.device = device
      self.n_agents_by_net = n_agents_by_net
      self.out_features = encoding_size
      self.n_inputs_by_node_class = n_inputs_by_node_class
      self.use_rnn = use_rnn
      self.activation = get_activation(activation)
      self.layer1 = th.nn.ModuleList()
      self.layer2 = th.nn.ModuleList()
      self.hidden_states = []

      for in_size in n_inputs_by_node_class:
         self.layer1.append(th.nn.Linear(in_size, encoding_size))

         l2 = th.nn.LSTMCell(encoding_size, encoding_size) if use_rnn else th.nn.Linear(
             encoding_size, encoding_size)

         self.layer2.append(l2)

      def init_xavier(m):
         if type(m) in (th.nn.Linear, th.nn.LSTMCell):
            th.nn.init.xavier_uniform_(m.weight, gain=th.nn.init.calculate_gain('sigmoid'))

      for nets in [self.layer1, self.layer2]:
         if nets is not None:
            for net in nets:
               net.apply(init_xavier)

   def init_hidden(self, batch_size: int):
      assert isinstance(self.layer2[0], th.nn.LSTMCell)

      self.hidden_states = []
      for i in range(len(self.n_agents_by_net)):
         hs = th.zeros(self.n_agents_by_net[i] * batch_size,
                       self.layer2[i].hidden_size,
                       device=self.device)
         cs = th.zeros(self.n_agents_by_net[i] * batch_size,
                       self.layer2[i].hidden_size,
                       device=self.device)
         self.hidden_states.append((hs, cs))

   def apply_net(self, x: th.tensor, index: int):
      # remove singleton dim
      x.squeeze_()
      if x.ndim == 1:
         x = x.unsqueeze(0)
      assert x.ndim == 2, "only agent dim and feature dim here!"

      l1 = self.layer1[index]
      l2 = self.layer2[index]

      x = self.activation(l1(x))

      if self.use_rnn:
         hidden, cell = self.hidden_states[index]
         x, cell = l2(x, (hidden, cell))
         self.hidden_states[index] = (x, cell)
      else:
         x = l2(x)

      return self.activation(x)

   def forward(self, x: dict, node_type: th.tensor):
      """Encode node features

      :param x: Dictionary containing node classes as keys and tensors with their respective features as values
      :type x: dict
      :param node_type: tensor containing the class of each node
      :type node_type: th.tensor
      :return: a tensor witht he encoded features of all nodes
      :rtype: th.tensor
      """
      # create tensor to hold the encoded results
      X = th.empty(node_type.size(0), self.out_features, device=self.device)

      for nt in node_type.unique().tolist():
         node_mask = (node_type == nt)  # grab nodes of that type
         enc = self.apply_net(x[nt], nt)  # apply corresponding layer to input
         X[node_mask] = enc  # put outputs in their corresponding places

      return X
