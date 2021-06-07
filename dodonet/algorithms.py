import math
import random
from abc import ABC
from datetime import datetime
from enum import Enum
from os.path import isfile

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data

from dodonet.encoding import isin
from dodonet.nn import (HeteroMAGNet, HeteroMAGNetA2CMLP, HeteroMAGNetBaseLayer, HeteroMAGQNet)
from dodonet.nn.modules.graph import GATModule
from dodonet.rl.replay import Transition


class MultiAgentDRLAlgorithm(ABC):

    class Policy(Enum):
        RANDOM = 'Random'
        BOLTZMANN = 'Boltzmann'
        GREEDY = 'Greedy'
        EGREEDY = 'e-Greedy'
        EGREEDY_DECAY = 'e-Greedy with Decay'

    def __init__(
        self,
        checkpoint_file: str,
        summary_writer: SummaryWriter,
        policy: str,
        max_num_steps: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        trr_coef: float,
        checkpoint_save_secs: int,
        graph_layer_type: str,
        full_agent_communication: bool,
        full_receptive_field: bool,
        gat_n_heads: int,
        gat_average_last: bool,
        rgcn_n2_relations: bool,
        rgcn_num_bases: int,
        device,
        node_types: list,
        agent_types: list,
        features_by_node_type: list,
        actions_by_node_type: list,
        training_mode: bool,
        gamma: float = 1,
        eps_start: float = 1,
        eps_end: float = 1
    ):
        """Base agent that employs the HMAGNet as its policy net

        :param checkpoint_file: name of the ``.pth`` checkpoint file to save net and optimizer states
        :type checkpoint_file: str
        :param summary_writer: initialized writer for TenSorBoard
        :type summary_writer: SummaryWriter
        :param max_num_steps: maximum number of training steps, defaults to 1e6
        :type max_num_steps: int, optional
        :param batch_size: batch size, defaults to 32
        :type batch_size: int, optional
        :param lr: learning rate of the neural network, defaults to 1e-4
        :type lr: float, optional
        :param weight_decay: L2 regularization factor, defaults to 1e-5
        :type weight_decay: float, optional
        :param trr_coef: coefficient for temporal relation regularization, defaults to 1e-3
        :type trr_coef: float, optional
        :param checkpoint_save_secs: how many wallclock seconds to wait between checkpoint saves, defaults to 180
        :type checkpoint_save_secs: int, optional
        :param graph_layer_type: GCN, GAT or RGCN, defaults to 'GAT'
        :type graph_layer_type: str, optional
        :param full_receptive_field: the graph module outputs the output of all layers, instead of only the last one, defaults to True
        :type full_receptive_field: bool, optional
        :param gat_n_heads: number of attention heads in the GAT module, defaults to 3
        :type gat_n_heads: int, optional
        :param gat_average_last: average the last GAT layer, instead of concatenating, defaults to False
        :type gat_average_last: bool, optional
        :param rgcn_n2_relations: use one relation for each pair of agent-type/node-type in RGCN, else use one relation per agent-type, defaults to True
        :type rgcn_n2_relations: bool, optional
        :param rgcn_num_bases: number of basis matrices for RGCN layers, defaults to 1
        :type rgcn_num_bases: int, optional
        :param device: which PyTorch device to use, defaults to None
        :param node_types: [description], defaults to None
        :type node_types: list, optional
        :param agent_types: [description], defaults to None
        :type agent_types: list, optional
        :param features_by_node_type: [description], defaults to None
        :type features_by_node_type: list, optional
        :param actions_by_node_type: [description], defaults to None
        :type actions_by_node_type: list, optional
        """
        self.num_unit_types = len(set(node_types))

        assert self.num_unit_types == len(features_by_node_type) == len(actions_by_node_type)
        assert all([at in node_types for at in agent_types])

        self.checkpoint = checkpoint_file
        self.summary_writer = summary_writer
        self.device = device
        self.training_mode = training_mode
        self.max_num_steps = max_num_steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.trr_coef = trr_coef
        self.checkpoint_save_secs = checkpoint_save_secs
        self.graph_layer_type = HeteroMAGNetBaseLayer.GraphLayerType[graph_layer_type]
        self.gat_n_heads = gat_n_heads
        self.gat_average_last = gat_average_last
        self.rgcn_n2_relations = rgcn_n2_relations
        self.rgcn_num_bases = rgcn_num_bases
        self.full_agent_communication = full_agent_communication
        self.full_receptive_field = full_receptive_field
        self.policy = MultiAgentDRLAlgorithm.Policy[policy]
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = self.max_num_steps * .1

        self.agent_types = torch.tensor(agent_types, dtype=torch.long, device=self.device)
        self.n_agents = sum(node_types.count(agent_type) for agent_type in agent_types)
        self.n_actions_agents = [actions_by_node_type[nt] for nt in node_types if nt in agent_types]

        self.total_steps = 0
        self.optim_steps = 0
        self.n_episodes = 0
        self.current_state = None
        self.next_state = None
        self.last_save = None
        self.n_enemies = None
        self.policy_net: HeteroMAGNet = None
        self.optimizer: optim.optimizer.Optimizer = None

    def gen_state(self, *args, **kwargs) -> Data:
        """Generates a graph state for the environment, given the arguments

        :raises NotImplementedError: must be implemented by a wrapper class
        :return: state in graph form
        :rtype: Data
        """
        raise NotImplementedError

    def _optimize(self, trans_or_trajs) -> tuple:
        """The core of the optimization procedure, responsible for calculating the loss (in the HMAGNet case, the sum of class losses for each output network), temporal relation regulation (pass 0 to not use it) and optional TD errors, used by some methods to prioritize the experience replay (pass None if not used).

        Note that the returned loss and TRR will be added and ``backward()`` will be called in the resulting scalar.

        :param trans_or_trajs: the data used for one optimization step (e.g. a batch of transitions or trajectories)
        :raises NotImplementedError: Must be implemented by subclasses, this class only calls this method and uses the returned values for logging and optimization
        :return: tuple containing loss as a scalar tensor, trr as a scalar tensor, TD errors of states present in ``trans_or_trajs``
        :rtype: tuple
        """
        raise NotImplementedError

    def log_net_weights(self):
        """Log the policy network weights as histograms/distributions"""
        for tag, value in self.policy_net.named_parameters():
            tag = tag.replace('.', '/')
            self.summary_writer.add_histogram(tag, value.data.cpu().numpy(), self.optim_steps)
            self.summary_writer.add_histogram(
                tag + '/grad',
                value.grad.data.cpu().numpy(), self.optim_steps
            )

    def act(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def add_all_agent_edges(self, state: Data) -> Data:
        """Add edges between all agent nodes

        :param state: a graph state
        :type state: Data
        :return: the same graph state, with edges between all agents added
        :rtype: Data
        """
        agent_nodes = isin(state.node_type, self.agent_types)
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

    @staticmethod
    def random_policy(valid_actions: list, device) -> torch.Tensor:
        """Randomly selects an action for each agent, among the valid ones.

        :param valid_actions: list of lists containing boolean masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: list
        :return: 1D tensor containing the index of the selected action for each agent. Selected actions must be valid for application in the current state.
        :rtype: torch.Tensor
        """
        n_agents = len(valid_actions)
        actions = torch.empty(n_agents, dtype=torch.int, device=device)
        for agent_id in range(n_agents):
            av_act = valid_actions[agent_id].nonzero().squeeze()

            if len(av_act.shape) == 0:
                actions[agent_id] = av_act
            else:
                chosen_action = torch.randint(av_act.shape[0], (1, ))
                actions[agent_id] = av_act[chosen_action]

        return actions

    @staticmethod
    def greedy_policy(network: HeteroMAGNet, valid_actions: list, state: Data) -> torch.Tensor:
        """Use the model contained in the policy network to select the best actions for all
        agents, given the current state.

        :param valid_actions: list of boolean tensors, which are used as masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: list
        :param state: the state to be evaluated
        :type state: torch_geometric.data.Data
        :return: 1D tensor containing the actions chosen for all agents
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            network.eval()
            output_dict = network(state)
            network.train()

        n_agents = sum(output_dict[agent_type][0].size(0) for agent_type in output_dict)
        actions = torch.empty(n_agents, dtype=torch.int, device=network.device)

        for agent_type in output_dict:
            agent_ids, agent_outputs = output_dict[agent_type]

            for i, agent_id in enumerate(agent_ids):
                inv_actions_agent = ~valid_actions[agent_id]
                agent_output = agent_outputs[i]

                agent_output[inv_actions_agent] = float('-inf')
                best_actions = agent_output.argsort(descending=True)

                actions[agent_id] = best_actions[0]

        return actions

    @staticmethod
    def _boltzmann_policy(output_dict: dict, valid_actions: list, device, temperature: float = 1.):
        """Selects actions for agents according to a Boltzmann policy, which applies a softmax function on the Q values of each individual agent and samples an action from the resulting probabilities. An optional temperature value :math:`\\tau \\in (0, 1]` can be used to control the spread of the distribution, making the final probability of selecting an action for a given agent equal to

        .. math::

            P(a_i) = \\frac{\\exp(Q_{a_i}/\\tau)}{\\sum_{a \\in A}\\exp(Q_a/\\tau)}.

        :param valid_actions: list of boolean tensors, which are used as masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: list
        :param temperature: [description]
        :type temperature: float
        """

        n_agents = len(valid_actions)
        actions = torch.empty(n_agents, dtype=torch.int, device=device)
        dists = [None] * n_agents

        for agent_type in output_dict:
            agent_ids, agent_outputs = output_dict[agent_type]

            for i, agent_id in enumerate(agent_ids):
                agent_output = agent_outputs[i]

                final_probs = torch.zeros(agent_output.shape, device=device)
                # apply softmax only in the valid actions, otherwise 0 prob of invalid actions wouldn't be 0 anymore
                final_probs[valid_actions[agent_id]
                            ] = (agent_output[valid_actions[agent_id]] / temperature).softmax(-1)

                dists[agent_id] = torch.distributions.Categorical(probs=final_probs)
                actions[agent_id] = dists[agent_id].sample()

        return actions, dists

    @staticmethod
    def stochastic_policy(output_dict, valid_actions, device):
        return MultiAgentDRLAlgorithm._boltzmann_policy(output_dict, valid_actions, device)

    @staticmethod
    def boltzmann_policy(
        network: HeteroMAGNet,
        valid_actions: list,
        state: Data,
        temperature: float = 1.
    ) -> torch.Tensor:
        """Selects actions for agents according to a Boltzmann policy, which applies a softmax function on the Q values of each individual agent and samples an action from the resulting probabilities. An optional temperature value :math:`\\tau \\in (0, 1]` can be used to control the spread of the distribution, making the final probability of selecting an action for a given agent equal to

        .. math::

            P(a_i) = \\frac{\\exp(Q_{a_i}/\\tau)}{\\sum_{a \\in A}\\exp(Q_a/\\tau)}.

        :param valid_actions: list of boolean tensors, which are used as masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: list
        :param state: the state to be evaluated
        :type state: torch_geometric.data.Data
        :param temperature: [description]
        :return: 1D tensor containing the actions chosen for all agents
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            network.eval()
            output_dict = network(state)
            network.train()

        return MultiAgentDRLAlgorithm._boltzmann_policy(
            output_dict, valid_actions, network.device, temperature
        )[0]

    @staticmethod
    def egreedy_policy(
        network: HeteroMAGNet, valid_actions: list, state: Data, epsilon: float
    ) -> torch.Tensor:
        """Selects an action according to an epsilon-greedy policy. Given a uniform random
        value :math:`r \\sim U(0, 1)`, use :py:func:`greedy_policy` if :math:`r \\leq \\epsilon`,
        else use :py:func:`random_policy`.

        :param valid_actions: list of boolean tensors, which are used as masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: list
        :param state: the state to be evaluated
        :type state: torch_geometric.data.Data
        :param epsilon: :math:`\\epsilon`
        :type epsilon: float
        :return: 1D tensor containing the actions chosen for all agents
        :rtype: torch.Tensor
        """
        sample = random.random()
        greedy = state is not None and sample > epsilon

        if greedy:
            actions = MultiAgentDRLAlgorithm.greedy_policy(network, valid_actions, state)
        else:
            actions = MultiAgentDRLAlgorithm.random_policy(valid_actions, network.device)

        return actions

    @staticmethod
    def egreedy_decay_policy(
        network: HeteroMAGNet, valid_actions: list, state: Data, step_num: int, eps_start, eps_end,
        eps_decay
    ) -> torch.Tensor:
        """Use :py:func:`egreedy_policy`, but calculate :math:`\\epsilon` according to an
        exponential decay strategy. The probability of choosing a random action starts at
        ``self.eps_start`` and will decay towards ``self.eps_end``. ``self.eps_decay``
        controls the rate of the decay.

        :param valid_actions: list of boolean tensors, which are used as masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: torch.Tensor
        :param state: the state to be evaluated
        :type state: torch_geometric.data.Data
        :param step_num: current training step
        :type step_num: int
        :return: 1D tensor containing the actions chosen for all agents
        :rtype: torch.Tensor
        """
        epsilon = eps_end + (eps_start - eps_end) * math.exp(-1. * step_num / eps_decay)

        return MultiAgentDRLAlgorithm.egreedy_policy(network, valid_actions, state, epsilon)

    @staticmethod
    def _get_trr(
        network: HeteroMAGNet, states: list, next_states: list, non_final_next_states_indices: list,
        trr_coef: float
    ):
        """Calculate temporal relation regularization. Attention weights from the last layer of the communication module of the policy network after processing state :math:`s` are compared against the same attention weights after processing state :math:`s'`. Comparison is done using the Kullback Leibler divergence. The larger the divergence, the larger the regularization factor.

        :param states: list of current states
        :type states: list
        :param next_states: list of next states
        :type next_states: list
        :param non_final_next_states_indices: indices of non final next states
        :type non_final_next_states_indices: list
        :raises NotImplementedError: if the communication layer is not a GAT layer
        :return: TRR factor
        :rtype: torch.Tensor
        """
        if trr_coef == 0:
            return 0

        if type(network.base_layer.relational_layer) != GATModule:
            raise NotImplementedError(
                "Only GAT layer type has attention weights necessary to calculate temporal relation regularization"
            )
        # turn the policy net to eval mode
        with torch.no_grad():
            network.eval()

            # run the batch of current states and get attention from last layer
            valid_states = Batch.from_data_list([states[i] for i in non_final_next_states_indices])
            network(valid_states)
            edge_index = network.base_layer.relational_layer.attention_indices
            att_states = network.base_layer.relational_layer.last_layer_attention

            # do the same with the batch of next non-final states
            non_final_next_states_batch = Batch.from_data_list(
                [next_states[i] for i in non_final_next_states_indices]
            )
            network(non_final_next_states_batch)
            next_edge_index = network.base_layer.relational_layer.attention_indices
            att_next_states = network.base_layer.relational_layer.last_layer_attention

        network.train()

        # create two dense adjacency matrices to hold the organized attention weights
        # start them with zeros (absence of edges)
        final_dist1 = torch.zeros(
            (
                valid_states.num_nodes, valid_states.num_nodes,
                network.base_layer.relational_layer.n_heads
            ),
            device=network.device
        )
        final_dist2 = torch.zeros(
            (
                non_final_next_states_batch.num_nodes, non_final_next_states_batch.num_nodes,
                network.base_layer.relational_layer.n_heads
            ),
            device=network.device
        )

        # put attention weights into dense matrices
        for i, edge in enumerate(edge_index.T):
            final_dist1[edge[0], edge[1], :] = att_states[i]
        for i, edge in enumerate(next_edge_index.T):
            final_dist2[edge[0], edge[1], :] = att_next_states[i]

        # order the tensor so that all graphs/states are kept in dimension 0
        final_dist1 = final_dist1.view(valid_states.num_graphs, -1)
        final_dist2 = final_dist2.view(non_final_next_states_batch.num_graphs, -1)

        # input batches must sum to 1 and be a log (logprob)
        final_dist1 = final_dist1.log_softmax(0)
        # target batches must sum to 1
        final_dist2 = final_dist2.softmax(0)
        # calculate kl_div
        return trr_coef * F.kl_div(final_dist1, final_dist2, reduction='batchmean')

    def optimize(self, trainer):
        if not self.training_mode:
            raise RuntimeError("Optimization not allowed when not in training mode")

        self.optim_steps += 1
        # Transpose the batch (see https://stackoverflow.com/a/19343 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        loss, trr = self._optimize(trainer)

        total_loss = loss + trr

        self.summary_writer.add_scalar('losses/sum_class_loss', loss, self.optim_steps)
        if trr != 0:
            self.summary_writer.add_scalar('losses/trr', trr, self.optim_steps)
            self.summary_writer.add_scalar('losses/total_loss', total_loss, self.optim_steps)

        # Optimize the model
        # zero the gradients of the neural network,
        # otherwise they are accumulated
        self.optimizer.zero_grad()
        # calculate the gradients wrt. the loss function
        total_loss.backward()

        # NOTE Mnih 2015, clips the gradients instead
        # of using the Huber loss
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # perform an optimization step
        self.optimizer.step()

        if self.optim_steps % 250 == 0:
            self.log_net_weights()

    def maybe_save_checkpoint(self, net_state_dicts: dict, step_num, training_datetime):
        if self.last_save is None:
            self.last_save = datetime.now()
        elif (datetime.now() - self.last_save).total_seconds() >= self.checkpoint_save_secs:
            print("""Training step: {}
                Saving checkpoint...""".format(step_num))

            variables = {
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps': step_num,
                'training_datetime': training_datetime,
                'optim_steps': self.optim_steps,
                'n_episodes': self.n_episodes
            }

            variables = {**net_state_dicts, **variables}

            torch.save(variables, self.checkpoint)
            self.last_save = datetime.now()


class MultiAgentDQN(MultiAgentDRLAlgorithm, ABC):

    def __init__(
        self,
        checkpoint_file: str,
        summary_writer: SummaryWriter,
        policy: str,
        max_num_steps: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        trr_coef: float,
        checkpoint_save_secs: int,
        graph_layer_type: str,
        full_agent_communication: bool,
        full_receptive_field: bool,
        coalesce_state_obs: bool,
        gat_n_heads: int,
        gat_average_last: bool,
        rgcn_n2_relations: bool,
        rgcn_num_bases: int,
        device,
        node_types: list,
        agent_types: list,
        features_by_node_type: list,
        actions_by_node_type: list,
        training_mode: bool,
        gamma: float,
        eps_start: float,
        eps_end: float,
        target_update: int,
        encoding_hidden_sizes=None,
        encoding_output_size=64,
        graph_hidden_sizes=None,
        graph_output_size=64,
        action_hidden_sizes=None
    ):
        super().__init__(
            checkpoint_file, summary_writer, policy, max_num_steps, batch_size, lr, weight_decay,
            trr_coef, checkpoint_save_secs, graph_layer_type, full_agent_communication,
            full_receptive_field, gat_n_heads, gat_average_last, rgcn_n2_relations, rgcn_num_bases,
            device, node_types, agent_types, features_by_node_type, actions_by_node_type,
            training_mode, gamma, eps_start, eps_end
        )
        if encoding_hidden_sizes is None:
            encoding_hidden_sizes = [128, 128]
        if graph_hidden_sizes is None:
            graph_hidden_sizes = [128, 128, 128]
        if action_hidden_sizes is None:
            action_hidden_sizes = [128, 128]

        self.target_update = target_update
        self.target_net: HeteroMAGNet
        # either use SmoothL1Loss (Huber loss) to stabilize gradients
        # or MSELoss (quadratic loss) and clip gradients
        self.criterion = torch.nn.MSELoss()

        self.policy_net = HeteroMAGQNet(
            self.num_unit_types, agent_types, features_by_node_type, encoding_hidden_sizes,
            encoding_output_size, graph_hidden_sizes, graph_output_size, action_hidden_sizes,
            actions_by_node_type, self.device, self.graph_layer_type, self.full_receptive_field,
            self.gat_n_heads, self.gat_average_last, self.rgcn_n2_relations, self.rgcn_num_bases,
            True
        ).to(self.device)
        self.target_net = HeteroMAGQNet(
            self.num_unit_types, agent_types, features_by_node_type, encoding_hidden_sizes,
            encoding_output_size, graph_hidden_sizes, graph_output_size, action_hidden_sizes,
            actions_by_node_type, self.device, self.graph_layer_type, self.full_receptive_field,
            self.gat_n_heads, self.gat_average_last, self.rgcn_n2_relations, self.rgcn_num_bases,
            coalesce_state_obs
        ).to(self.device)

        nn_trainable_parameters = sum(
            p.numel() for p in self.policy_net.parameters() if p.requires_grad
        )
        print('Neural network has {} trainable parameters'.format(nn_trainable_parameters))

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # NOTE the default value of momentum is 0, but the DQN paper uses 0.95
        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.95)
        # optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

        # instantiate step learning scheduler class
        # gamma = decaying factor
        # after every epoch, new_lr = lr*gamma
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.96)

        if isfile(self.checkpoint):
            print('Loading from checkpoint...')
            checkpoint = torch.load(self.checkpoint)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps = checkpoint['total_steps']
            self.optim_steps = checkpoint['optim_steps']
            self.n_episodes = checkpoint['n_episodes']
        elif not self.training_mode:
            raise FileNotFoundError(
                "Checkpoint file {} does not exists. It is necessary in order to play a game.".
                format(self.checkpoint)
            )
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            hparams = {
                'trr_coef': self.trr_coef,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'max_num_steps': self.max_num_steps,
                'optim_steps': self.optim_steps,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'eps_start': self.eps_start,
                'eps_end': self.eps_end,
                'eps_decay': self.eps_decay,
                'target_update': self.target_update,
                'num_agent_types': len(agent_types),
                'num_node_types': self.num_unit_types,
                'encoding_n_hidden_layers': len(encoding_hidden_sizes),
                'encoding_n_hidden_neurons': encoding_hidden_sizes[0],
                'encoding_output_size': encoding_output_size,
                'graph_layer_type': str(self.policy_net.base_layer.graph_layer_type),
                'policy': str(self.policy),
                'graph_n_hidden_layers': len(graph_hidden_sizes),
                'graph_n_hidden_neurons': graph_hidden_sizes[0],
                'graph_output_size': graph_output_size,
                'action_n_hidden_layers': len(action_hidden_sizes),
                'action_n_hidden_neurons': action_hidden_sizes[0],
                'gat_n_heads': self.gat_n_heads,
                'gat_average_last': self.gat_average_last,
                'rgcn_n2_relations': self.rgcn_n2_relations,
                'full_receptive_field': self.full_receptive_field,
                'full_agent_communication': self.full_agent_communication
            }
            self.summary_writer.add_hparams(hparams, {})

            # NOTE PyG layers don't work with TensorBoard, so I can't see the graph
            # self.summary_writer.add_graph(self.policy_net)
            # self.summary_writer.add_graph(self.target_net)
            # self.summary_writer.close()

        self.policy_net.train(self.training_mode)
        self.target_net.eval()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(
        self, state: Data, valid_actions: list, step_num: int, epsilon: int = 1
    ) -> torch.Tensor:
        """Select an action according to the selected policy

        :param state: the state to be evaluated
        :type state: torch_geometric.data.Data
        :param valid_actions: list of lists containing boolean masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: list
        :param step_num: current training step
        :type step_num: int
        :param epsilon: :math:`\\epsilon`, defaults to 1
        :type epsilon: float, optional
        :return: 1D tensor containing the actions chosen for all agents
        :rtype: torch.Tensor
        """
        for i in range(len(valid_actions)):
            valid_actions[i] = torch.tensor(valid_actions[i], dtype=torch.bool)
            while len(valid_actions[i].size()) < 1:
                valid_actions[i].unsqueeze_(-1)

        network = self.policy_net

        if self.policy == MultiAgentDRLAlgorithm.Policy.RANDOM:
            actions = self.random_policy(valid_actions, self.device)
        if self.policy == MultiAgentDRLAlgorithm.Policy.BOLTZMANN:
            actions = self.boltzmann_policy(network, valid_actions, state)
        if self.policy == MultiAgentDRLAlgorithm.Policy.GREEDY:
            actions = self.greedy_policy(network, valid_actions, state)
        if self.policy == MultiAgentDRLAlgorithm.Policy.EGREEDY:
            actions = self.egreedy_policy(network, valid_actions, state, epsilon)
        if self.policy == MultiAgentDRLAlgorithm.Policy.EGREEDY_DECAY:
            actions = self.egreedy_decay_policy(
                network, valid_actions, state, step_num, self.eps_start, self.eps_end,
                self.eps_decay
            )

        return actions

    def _optimize(self, trainer):
        # Transpose the batch (see https://stackoverflow.com/a/19343 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        assert len(
            trainer.memory
        ) >= trainer.min_memory_size, "Replay memory hasn't reached the minimum size for an optimization step"

        transitions, weights, batch_idxes = trainer.memory.sample(
            trainer.batch_size, beta=trainer.replay_buffer_beta
        )

        batch = Transition(*zip(*transitions))

        # STEP 1: break down the batch
        ##########################################################
        state_batch = Batch.from_data_list(batch.state)
        action_batch = torch.stack(batch.action).to(self.device).long().flatten()
        reward_batch = torch.stack(batch.reward).to(self.device).flatten()

        # get the indices of non-final next states and create a batch
        non_final_next_states_indices = torch.tensor([s is not None for s in batch.next_state]
                                                     ).nonzero().squeeze()
        non_final_next_states = Batch.from_data_list(
            [batch.next_state[i] for i in non_final_next_states_indices]
        )
        non_terminal_nodes = torch.cat(
            list(
                torch.tensor([t.next_state is not None] * self.n_agents, dtype=torch.bool)
                for t in transitions
            )
        )

        # STEP 2: apply the networks to current and next states
        ##########################################################
        # compute Q(s_t) using the model
        state_dict = self.policy_net(state_batch)

        # Compute V(s_{t+1}) for all non-final next states,
        # using the "older" target_net and reshape it by batch_size x n_agents x n_actions
        next_state_dict = self.target_net(non_final_next_states)

        # state_values = state_values.view(self.batch_size, self.n_agents, self.n_actions)
        # q_snext_non_final = q_snext_non_final.view(
        #     non_final_next_states.num_graphs, self.n_agents, self.n_actions
        # )

        # STEP 3: process each agent class separately
        ##########################################################
        loss = torch.tensor(0, device=self.device)
        td_errors = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        # flattened_action_batch = action_batch.flatten()

        for agent_type in state_dict:
            # CURRENT STATE         32                22
            # agent_outputs is n_batches x (n_actions x n_agents)
            agent_ids, agent_outputs = state_dict[agent_type]

            # agent_outputs = agent_outputs.view(self.batch_size, -1, self.n_actions)
            actions_of_group = action_batch[agent_ids].unsqueeze(1)
            reward_of_group = reward_batch[agent_ids]

            # select the columns of actions taken by the agents in each state
            state_action_values_by_class = torch.gather(agent_outputs, 1,
                                                        actions_of_group).squeeze()

            # NEXT STATE
            next_agent_ids, next_agent_outputs = next_state_dict[agent_type]
            # next_agent_outputs = next_agent_outputs.view(
            #     non_final_next_states_indices.size(0), -1, self.n_actions
            # )

            # assert torch.eq(agent_ids, next_agent_ids)

            t1 = non_terminal_nodes[agent_ids]
            t1 = t1.nonzero().squeeze()

            # initialize Q(s_{t+1}, a) with all zeroes
            next_state_action_values_by_class = torch.zeros_like(
                state_action_values_by_class, device=self.device
            )

            # get the max q(s_{t+1},a) for each agent
            # replace in the tensor
            # NOTE maybe need to call detach()
            ububu = next_agent_outputs.max(1)[0]
            next_state_action_values_by_class[t1] = ububu

            # Compute y, the expected Q values
            expected_state_action_values_by_class = (
                next_state_action_values_by_class * self.gamma
            ) + reward_of_group

            # loss for this class of agents
            loss_class = self.criterion(
                state_action_values_by_class, expected_state_action_values_by_class
            )
            # accumulate loss
            loss = loss + loss_class

            self.summary_writer.add_scalar(
                'losses/class_{}'.format(agent_type), loss_class, self.optim_steps
            )

            # calculate td error for agents individually,
            # join them in batches, take the mean of the batch
            # and accumulate for each class of agent
            td_error_class = torch.abs(
                state_action_values_by_class - expected_state_action_values_by_class
            ).view(self.batch_size, -1).mean(1)
            td_errors += td_error_class

        # calculate temporal relation regularization from Jiang 2020
        trr = self._get_trr(
            self.policy_net, batch.state, batch.next_state, non_final_next_states_indices,
            self.trr_coef
        )

        # NOTE here is where we'd need to add prioritized_replay_eps to the td_errors, before updating priorities

        trainer.memory.update_priorities(batch_idxes, td_errors.tolist())

        return loss, trr


class MultiAgentActorCritic(MultiAgentDRLAlgorithm, ABC):

    @staticmethod
    def discount_with_dones(rewards, dones, gamma):
        """
        Apply the discount value to the reward, where the environment is not done
        :param rewards: ([float]) The rewards
        :param dones: ([bool]) Whether an environment is done or not
        :param gamma: (float) The discount value
        :return: ([float]) The discounted rewards
        """
        discounted = []
        ret = 0  # Return: discounted reward
        for reward, done in zip(rewards[::-1], dones[::-1]):
            ret = reward + gamma * ret * (1. - done)  # fixed off by one bug
            discounted.append(ret)
        return discounted[::-1]

    @staticmethod
    def compute_returns(next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def __init__(
        self,
        checkpoint_file: str,
        summary_writer: SummaryWriter,
        policy: str,
        max_num_steps: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        trr_coef: float,
        checkpoint_save_secs: int,
        graph_layer_type: str,
        full_agent_communication: bool,
        full_receptive_field: bool,
        coalesce_state_obs: bool,
        gat_n_heads: int,
        gat_average_last: bool,
        rgcn_n2_relations: bool,
        rgcn_num_bases: int,
        device,
        node_types: list,
        agent_types: list,
        features_by_node_type: list,
        actions_by_node_type: list,
        training_mode: bool,
        gamma: float,
        eps_start: float,
        eps_end: float,
        encoding_hidden_sizes=[128, 128],
        encoding_output_size=64,
        graph_hidden_sizes=[128, 128, 128],
        graph_output_size=64,
        action_hidden_sizes=[128, 128]
    ):
        super().__init__(
            checkpoint_file, summary_writer, policy, max_num_steps, batch_size, lr, weight_decay,
            trr_coef, checkpoint_save_secs, graph_layer_type, full_agent_communication,
            full_receptive_field, gat_n_heads, gat_average_last, rgcn_n2_relations, rgcn_num_bases,
            device, node_types, agent_types, features_by_node_type, actions_by_node_type,
            training_mode, gamma, eps_start, eps_end
        )
        self.policy_net = HeteroMAGNetA2CMLP(
            self.num_unit_types, agent_types, features_by_node_type, encoding_hidden_sizes,
            encoding_output_size, graph_hidden_sizes, graph_output_size, action_hidden_sizes,
            actions_by_node_type, self.device, self.graph_layer_type, self.full_receptive_field,
            self.gat_n_heads, self.gat_average_last, self.rgcn_n2_relations, self.rgcn_num_bases,
            coalesce_state_obs
        ).to(self.device)

        # either use SmoothL1Loss (Huber loss) to stabilize gradients
        # or MSELoss (quadratic loss) and clip gradients
        # self.criterion = torch.nn.MSELoss()

        nn_trainable_parameters = sum(
            p.numel() for p in self.policy_net.parameters() if p.requires_grad
        )
        print('Neural network has {} trainable parameters'.format(nn_trainable_parameters))

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # NOTE the default value of momentum is 0, but the DQN paper uses 0.95
        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.95)
        # optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

        # instantiate step learning scheduler class
        # gamma = decaying factor
        # after every epoch, new_lr = lr*gamma
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.96)

        if isfile(self.checkpoint):
            print('Loading from checkpoint...')
            checkpoint = torch.load(self.checkpoint)
            self.policy_net.load_state_dict(checkpoint['actor_critic_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps = checkpoint['total_steps']
            self.optim_steps = checkpoint['optim_steps']
            self.n_episodes = checkpoint['n_episodes']
        elif not self.training_mode:
            raise FileNotFoundError(
                "Checkpoint file {} does not exists. It is necessary in order to play a game.".
                format(self.checkpoint)
            )
        else:
            hparams = {
                'trr_coef': self.trr_coef,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'max_num_steps': self.max_num_steps,
                'optim_steps': self.optim_steps,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'eps_start': self.eps_start,
                'eps_end': self.eps_end,
                'eps_decay': self.eps_decay,
                'num_agent_types': len(agent_types),
                'num_node_types': self.num_unit_types,
                'encoding_n_hidden_layers': len(encoding_hidden_sizes),
                'encoding_n_hidden_neurons': encoding_hidden_sizes[0],
                'encoding_output_size': encoding_output_size,
                'graph_layer_type': str(self.policy_net.base_layer.graph_layer_type),
                'policy': str(self.policy),
                'graph_n_hidden_layers': len(graph_hidden_sizes),
                'graph_n_hidden_neurons': graph_hidden_sizes[0],
                'graph_output_size': graph_output_size,
                'action_n_hidden_layers': len(action_hidden_sizes),
                'action_n_hidden_neurons': action_hidden_sizes[0],
                'gat_n_heads': self.gat_n_heads,
                'gat_average_last': self.gat_average_last,
                'rgcn_n2_relations': self.rgcn_n2_relations,
                'full_receptive_field': self.full_receptive_field,
                'full_agent_communication': self.full_agent_communication
            }
            self.summary_writer.add_hparams(hparams, {})

            # NOTE PyG layers don't work with TensorBoard, so I can't see the graph
            # self.summary_writer.add_graph(self.policy_net)
            # self.summary_writer.close()

        self.policy_net.train(self.training_mode)

    @staticmethod
    def act(agent_indices, probs, valid_actions, device):
        outputs = {}

        for agent_type in probs:
            outputs[agent_type] = (agent_indices[agent_type], probs[agent_type])

        return super().stochastic_policy(outputs, valid_actions, device)

    def _optimize(self, trainer) -> tuple:
        """The core of the optimization procedure, responsible for calculating the loss (in the HMAGNet case, the sum of class losses for each output network), temporal relation regulation (pass 0 to not use it) and optional TD errors, used by some methods to prioritize the experience replay (pass None if not used).

        Note that the returned loss and TRR will be added and ``backward()`` will be called in the resulting scalar.

        :raises NotImplementedError: Must be implemented by subclasses, this class only calls this method and uses the returned values for logging and optimization
        :return: tuple containing loss as a scalar tensor, trr as a scalar tensor, TD errors of states present in ``trans_or_trajs``
        :rtype: tuple
        """
        log_probs, values, rewards, done_mask, states = trainer.trans_or_trajs

        non_final_next_states_indices = (~torch.tensor(done_mask,
                                                       dtype=torch.bool)).nonzero().squeeze()

        log_probs, values, rewards, done_mask = torch.tensor(log_probs), torch.tensor(
            values
        ), torch.tensor(rewards), torch.tensor([1 - done for done in done_mask])

        returns = self.compute_returns(next_value, rewards, done_mask).detach()

        advantages = returns - values

        multi_agent_loss = torch.tensor(0)
        trr = torch.tensor(0)

        # calculate individual loss for each agent
        for i in range(values.size(0)):
            agent_log_probs = log_probs[i]
            # agent_values = values[i]
            # agent_rewards = rewards[i]
            # agent_returns = self.compute_returns(next_value, agent_rewards, done_mask).detach()
            # agent_advantage = agent_returns - agent_values

            agent_advantages = advantages[i]

            agent_actor_loss = -(agent_log_probs * agent_advantages.detach()).mean()
            agent_critic_loss = agent_advantages.pow(2).mean()
            agent_loss = agent_actor_loss + 0.5 * agent_critic_loss - 0.001 * trainer.entropies[i]

            multi_agent_loss = multi_agent_loss + agent_loss

            current_states = trainer.trans_or_trajs[states][:-1]
            next_states = trainer.trans_or_trajs[states][1:]
            trr = self._get_trr(
                self.policy_net, current_states, next_states, non_final_next_states_indices,
                self.trr_coef
            )

        return multi_agent_loss, trr, None
