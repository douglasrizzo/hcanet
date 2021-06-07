from collections import OrderedDict

from torch import nn


class HeteroMAGNetActionLayer(nn.Module):

    def __init__(self, node_type_input_sizes: list, node_type_output_sizes: list, device):
        super().__init__()
        assert len(node_type_input_sizes) == len(node_type_output_sizes)

        self.device = device
        self.node_type_output_sizes = node_type_output_sizes
        self.networks = nn.ModuleList()

    def forward(self, x: dict, node_type) -> dict:
        """"""
        raise NotImplementedError(
            "This class does not directly implement the forward() method, please instantiate one of its base classes"
        )


class MultiAgentMultiNetQLayer(HeteroMAGNetActionLayer):

    def __init__(
        self, node_type_input_sizes: list, n_hidden: list, node_type_output_sizes: list, device
    ):
        super().__init__(node_type_input_sizes, node_type_output_sizes, device)

        if n_hidden is None:
            n_hidden = []

        for i in range(len(node_type_input_sizes)):
            input_size = node_type_input_sizes[i]
            out_size = node_type_output_sizes[i]
            sizes = [input_size] + n_hidden + [out_size]
            net = nn.Sequential()

            for j in range(len(sizes) - 1):
                net.add_module('linear_{}'.format(j + 1), nn.Linear(sizes[j], sizes[j + 1]))

                # we only add nonlinearities to the hidden layers
                if j < len(sizes) - 2:
                    net.add_module('activation_{}'.format(j + 1), nn.Sigmoid())

            self.networks.append(net.to(self.device))

        def init_xavier(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('sigmoid'))

        for net in self.networks:
            if net is not None:
                net.apply(init_xavier)

    def forward(self, x: dict, node_type) -> dict:
        output_dict = {}

        # for each agent node type, retrieve their q-network and
        # calculate q(s) for all agents of the same type, as a batch
        # if there are multiple
        unique_types = node_type.unique().tolist()
        for nt in unique_types:
            node_indices = (node_type == nt).nonzero().squeeze(
            )  # get indices of nodes that belong to the current class

            action_net = self.networks[nt]  # get the correct network

            # apply it to the data, keep in a dict alongside node indices
            # need to transform key to int for a better hash
            output_dict[nt] = node_indices, action_net(x[nt])

        return output_dict


class MultiAgentMultiNetActorCriticLayer(HeteroMAGNetActionLayer):

    def __init__(
        self, node_type_input_sizes: list, n_hidden: list, node_type_output_sizes: list, device
    ):
        super().__init__(node_type_input_sizes, node_type_output_sizes, device)

        self.policy_heads = nn.ModuleList()
        self.value_heads = nn.ModuleList()

        if n_hidden is None:
            n_hidden = []

        for i in range(len(node_type_input_sizes)):
            input_size = node_type_input_sizes[i]
            out_size = node_type_output_sizes[i]
            sizes = [input_size] + n_hidden

            net = nn.Sequential()

            for j in range(len(sizes) - 1):
                net.add_module('linear_{}'.format(j + 1), nn.Linear(sizes[j], sizes[j + 1]))
                net.add_module('activation_{}'.format(j + 1), nn.Sigmoid())

            self.networks.append(net.to(self.device))

            self.policy_heads.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ('policy_linear', nn.Linear(n_hidden[-1], out_size)),
                            ('policy_softmax', nn.Softmax())
                        ]
                    )
                )
            )
            self.value_heads.append(
                nn.Sequential(OrderedDict([('value_linear', nn.Linear(n_hidden[-1], 1))]))
            )

        def init_xavier(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('sigmoid'))

        for net in self.networks + self.policy_heads + self.value_heads:
            if net is not None:
                net.apply(init_xavier)

    def forward(self, x: dict, node_type) -> dict:
        output_dict = {}

        # for each agent node type, retrieve their q-network and
        # calculate q(s) for all agents of the same type, as a batch
        # if there are multiple
        unique_types = node_type.unique().tolist()
        for nt in unique_types:
            node_indices = (node_type == nt).nonzero().squeeze(
            )  # get indices of nodes that belong to the current class

            # get the correct networks/heads
            action_net = self.networks[nt]
            policy_head = self.policy_heads[nt]
            value_head = self.value_heads[nt]

            # apply it to the data, keep in a dict alongside node indices
            # need to transform key to int for a better hash
            intermediate_value = action_net(x[nt])
            # NOTE it's important that the first two arguments are passed in this order, like the Q-network does
            output_dict[nt] = node_indices, policy_head(intermediate_value
                                                        ), value_head(intermediate_value)

        return output_dict
