import torch


class EncoderByType(torch.nn.Module):

    def __init__(self, node_type_input_sizes: list, n_hidden: list, encoding_size: int, device):
        super().__init__()
        self.device = device
        self.out_features = encoding_size
        self.node_type_input_sizes = node_type_input_sizes

        self.encodings = torch.nn.ModuleList()

        if n_hidden is None:
            n_hidden = []

        for in_size in node_type_input_sizes:
            sizes = [in_size] + n_hidden + [encoding_size]
            net = torch.nn.Sequential()

            for i in range(len(sizes) - 1):
                net.add_module(
                    'linear_{}'.format(i + 1),
                    torch.nn.Sequential(
                        torch.nn.Linear(sizes[i], sizes[i + 1]), torch.nn.Sigmoid()
                    )
                )

            self.encodings.append(net.to(self.device))

        def init_xavier(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(
                    m.weight, gain=torch.nn.init.calculate_gain('sigmoid')
                )

        for net in self.encodings:
            if net is not None:
                net.apply(init_xavier)

    def forward(self, x: dict, node_type: torch.Tensor):
        """Encode node features

        :param x: Dictionary containing node classes as keys and tensors with their respective features as values
        :type x: dict
        :param node_type: tensor containing the class of each node
        :type node_type: torch.Tensor
        :return: a tensor witht he encoded features of all nodes
        :rtype: torch.Tensor
        """
        # create tensor to hold the encoded results
        X = torch.empty(node_type.size(0), self.out_features, device=self.device)

        for nt in node_type.unique().tolist():
            node_indices = (node_type == nt).nonzero().squeeze()  # grab nodes of that type
            encoding_layer = self.encodings[nt]  # grab encoding layer for node type
            enc = encoding_layer(x[nt])  # apply layer to input
            X[node_indices] = enc  # put outputs in their corresponding places

        return X
