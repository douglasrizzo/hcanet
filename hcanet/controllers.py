import random
from abc import ABC
from datetime import datetime
from enum import Enum
from os.path import isfile
from typing import List

import torch as th
import torch.nn.functional as F
from torch import optim
from torch.cuda import device as torch_device
from torch_geometric.data import Batch, Data

import wandb

from .nn import HeteroMAGNet
from .nn.modules.action import HeteroMAGNetActionLayer
from .nn.modules.graph import GATModule
from .rl.epsilon_schedules import DecayThenFlatSchedule
from .rl.replay import Transition
from .training import TrainingConfig


class BaseController(ABC):
   def __init__(
      self,
      node_types,
      agent_types,
      features_by_node_type,
      actions_by_node_type,
      device: torch_device,
   ):
      self.num_unit_types = len(set(node_types))

      assert (self.num_unit_types == len(features_by_node_type) == len(actions_by_node_type))
      assert all([at in node_types for at in agent_types])

      self.device = device
      self.agent_types = th.tensor(agent_types, dtype=th.long, device=self.device)
      self.n_agents = sum(node_types.count(agent_type) for agent_type in agent_types)
      self.n_actions_agents = [actions_by_node_type[nt] for nt in node_types if nt in agent_types]

   def act(self, *args) -> th.tensor:
      raise NotImplementedError

   @staticmethod
   def _action_lists_to_tensors(actions: List[list]):
      for i in range(len(actions)):
         actions[i] = th.tensor(actions[i], dtype=th.bool)
         while len(actions[i].size()) < 1:
            actions[i].unsqueeze_(-1)

   @staticmethod
   def random_policy(valid_actions: list, device: torch_device) -> th.tensor:
      """Randomly selects an action for each agent, among the valid ones.

        :param valid_actions: list of lists containing boolean masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: list
        :return: 1D tensor containing the index of the selected action for each agent. Selected actions must be valid for application in the current state.
        :rtype: th.tensor
        """
      n_agents = len(valid_actions)
      actions = th.empty(n_agents, dtype=th.int, device=device)
      for agent_id in range(n_agents):
         av_act = valid_actions[agent_id].nonzero().squeeze()

         if len(av_act.shape) == 0:
            actions[agent_id] = av_act
         else:
            chosen_action = th.randint(av_act.shape[0], (1, ))
            actions[agent_id] = av_act[chosen_action]

      return actions


class RandomController(BaseController):
   def __init__(
      self,
      node_types,
      agent_types,
      features_by_node_type,
      actions_by_node_type,
      device: torch_device,
   ):
      super().__init__(node_types, agent_types, features_by_node_type, actions_by_node_type, device)

   def initialize(self):
      pass

   def act(self, valid_actions: list) -> th.tensor:
      BaseController._action_lists_to_tensors(valid_actions)
      return RandomController.random_policy(valid_actions, self.device)


class DRLController(BaseController, ABC):
   class Policy(Enum):
      RANDOM = "Random"
      BOLTZMANN = "Boltzmann"
      GREEDY = "Greedy"
      EGREEDY = "e-Greedy"
      EGREEDY_DECAY = "e-Greedy with Decay"

   def __init__(
      self,
      checkpoint_file: str,
      action_module: TrainingConfig.ActionModuleType,
      policy: str,
      max_num_steps: int,
      batch_size: int,
      optimizer: str,
      lr: float,
      weight_decay: float,
      rmsprop_alpha: float,
      rmsprop_eps: float,
      trr_coef: float,
      checkpoint_save_secs: int,
      graph_layer_type: str,
      share_encoding: bool,
      share_comms: bool,
      share_action: bool,
      full_agent_communication: bool,
      full_receptive_field: bool,
      gat_n_heads: int,
      gat_average_last: bool,
      rgcn_n2_relations: bool,
      rgcn_num_bases: int,
      rgcn_fast: bool,
      device: torch_device,
      node_types: list,
      agent_types: list,
      features_by_node_class: list,
      actions_by_agent_class: list,
      training_mode: bool,
      data_parallel: bool,
      act_encoding: str,
      act_comms: str,
      act_action: str,
      use_rnn_encoding: bool,
      use_rnn_action: bool,
      mixer: str,
      encoding_output_size=128,
      graph_module_sizes: list = None,
      action_hidden_size=128,
   ):
      super().__init__(
         node_types,
         agent_types,
         features_by_node_class,
         actions_by_agent_class,
         device,
      )
      self.checkpoint = checkpoint_file
      self.action_module = action_module
      self.training_mode = training_mode
      self.max_num_steps = max_num_steps
      self.batch_size = batch_size
      self.lr = lr
      self.weight_decay = weight_decay
      self.trr_coef = trr_coef
      self.checkpoint_save_secs = checkpoint_save_secs
      self.graph_layer_type = (HeteroMAGNet.GraphLayerType[graph_layer_type]
                               if graph_layer_type is not None else None)
      self.gat_n_heads = gat_n_heads
      self.gat_average_last = gat_average_last
      self.rgcn_n2_relations = rgcn_n2_relations
      self.rgcn_num_bases = rgcn_num_bases
      self.rgcn_fast = rgcn_fast
      self.full_agent_communication = full_agent_communication
      self.full_receptive_field = full_receptive_field
      self.policy = DRLController.Policy[policy]

      # self.total_steps = 0
      self.last_grad_log = 0
      self.n_episodes = 0
      self.current_state = None
      self.next_state = None
      self.last_save = None

      self.graph_module_sizes = graph_module_sizes
      self.action_hidden_size = action_hidden_size

      if self.graph_module_sizes is None:
         self.graph_module_sizes = [128, 128]

      if self.action_module == TrainingConfig.ActionModuleType.DQN:
         self.action_layer_type = HeteroMAGNetActionLayer.LayerType.DQN
      elif self.action_module == TrainingConfig.ActionModuleType.DDQN:
         self.action_layer_type = HeteroMAGNetActionLayer.LayerType.DDQN
      else:
         raise ValueError("invalid action layer type")

      self.policy_net = HeteroMAGNet(
         self.action_layer_type,
         share_encoding,
         share_comms,
         share_action,
         self.num_unit_types,
         agent_types,
         node_types,
         features_by_node_class,
         encoding_output_size,
         self.graph_module_sizes,
         self.action_hidden_size,
         actions_by_agent_class,
         act_encoding,
         act_comms,
         act_action,
         use_rnn_encoding,
         use_rnn_action,
         self.device,
         self.graph_layer_type,
         self.full_receptive_field,
         self.gat_n_heads,
         self.gat_average_last,
         self.rgcn_n2_relations,
         self.rgcn_num_bases,
         self.rgcn_fast,
         mixer,
      )

      self.policy_net = DRLController.maybe_parallel_net(self.policy_net,
                                                         self.device,
                                                         data_parallel)

      self.optimizer: optim.Optimizer
      if optimizer == "rmsprop":
         self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=self.lr,
            alpha=rmsprop_alpha,
            eps=rmsprop_eps,
            weight_decay=self.weight_decay,
         )
      elif optimizer == "adam":
         self.optimizer = optim.Adam(self.policy_net.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
      else:
         raise ValueError('Invalid optimizer "{}"'.format(optimizer))

   @staticmethod
   def maybe_parallel_net(net, device, data_parallel):
      if device.type == "cuda" and th.cuda.device_count() > 1 and data_parallel:
         print("Let's use", th.cuda.device_count(), "GPUs!")
         # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
         net = th.nn.DataParallel(net)

      net.to(device)

      return net

   def _optimize(self, trainer: TrainingConfig, memory) -> tuple:
      """The core of the optimization procedure, responsible for
        calculating the loss (in the HMAGNet case, the sum of class
        losses for each output network), temporal relation regulation
        (pass 0 to not use it) and optional TD errors, used by some
        methods to prioritize the experience replay (pass None if not used).

        Note that the returned loss and TRR will be added and ``backward()`` will be called in the resulting scalar.

        :raises NotImplementedError: Must be implemented by subclasses, this class only calls this method and uses the returned values for logging and optimization
        :return: tuple containing loss and trr, each one as a scalar tensor
        :rtype: tuple
        """
      raise NotImplementedError

   def act(self, *args) -> th.tensor:
      raise NotImplementedError

   @staticmethod
   def greedy_policy(q_values, valid_actions: list) -> th.tensor:
      """Use the model contained in the policy network to select the best actions for all agents, given the current state.

        :param valid_actions: list of boolean tensors, which are used as masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
        :type valid_actions: list
        :param state: the state to be evaluated
        :type state: torch_geometric.data.Data
        :return: 1D tensor containing the actions chosen for all agents
        :rtype: th.tensor
        """
      for agent_id in range(q_values.size(0)):
         inv_actions_agent = ~valid_actions[agent_id]
         q_values[agent_id, inv_actions_agent] = float("-inf")

      best_actions = q_values.argsort(descending=True, dim=1)
      actions = best_actions[:, 0]

      return actions.int()

   # @staticmethod
   # def _boltzmann_policy(values,
   #                       valid_actions: list,
   #                       device: torch_device,
   #                       temperature: float = 1.):
   #    """Selects actions for agents according to a Boltzmann policy, which applies a softmax function on the Q values of each individual agent and samples an action from the resulting probabilities. An optional temperature value :math:`\\tau \\in (0, 1]` can be used to control the spread of the distribution, making the final probability of selecting an action for a given agent equal to

   #      .. math::

   #          P(a_i) = \\frac{\\exp(Q_{a_i}/\\tau)}{\\sum_{a \\in A}\\exp(Q_a/\\tau)}.

   #      :param valid_actions: list of boolean tensors, which are used as masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
   #      :type valid_actions: list
   #      :param temperature: [description]
   #      :type temperature: float
   #      """

   #    n_agents = len(valid_actions)
   #    actions = th.empty(n_agents, dtype=th.int, device=device)
   #    dists = [None] * n_agents

   #    q_values, v_values = values

   #    for agent_id in range(n_agents):
   #       agent_output = q_values[agent_id]

   #       final_probs = th.zeros(agent_output.shape, device=device)
   #       # apply softmax only in the valid actions, otherwise 0 prob of invalid actions wouldn't be 0 anymore
   #       ava = valid_actions[agent_id]
   #       avp = agent_output[ava]
   #       final_probs[ava] = (avp / temperature).softmax(-1)
   #       # final_probs[valid_actions[agent_id]
   #       #             ] = (agent_output[valid_actions[agent_id]] / temperature).softmax(-1)

   #       dists[agent_id] = th.distributions.Categorical(probs=final_probs)
   #       actions[agent_id] = dists[agent_id].sample()

   #    return actions, dists

   # @staticmethod
   # def stochastic_policy(output_dict, valid_actions, device: torch_device):
   #    return MultiAgentDRLAlgorithm._boltzmann_policy(output_dict, valid_actions, device)

   # @staticmethod
   # def boltzmann_policy(output_dict: dict,
   #                      valid_actions: list,
   #                      device: torch_device,
   #                      temperature: float = 1.) -> th.tensor:
   #    """Selects actions for agents according to a Boltzmann policy, which applies a softmax function on the Q values of each individual agent and samples an action from the resulting probabilities. An optional temperature value :math:`\\tau \\in (0, 1]` can be used to control the spread of the distribution, making the final probability of selecting an action for a given agent equal to

   #      .. math::

   #          P(a_i) = \\frac{\\exp(Q_{a_i}/\\tau)}{\\sum_{a \\in A}\\exp(Q_a/\\tau)}.

   #      :param valid_actions: list of boolean tensors, which are used as masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
   #      :type valid_actions: list
   #      :param state: the state to be evaluated
   #      :type state: torch_geometric.data.Data
   #      :param temperature: [description]
   #      :return: 1D tensor containing the actions chosen for all agents
   #      :rtype: th.tensor
   #      """
   #    return MultiAgentDRLAlgorithm._boltzmann_policy(output_dict,
   #                                                    valid_actions,
   #                                                    device,
   #                                                    temperature)[0]

   @staticmethod
   def egreedy_policy(q_values: dict, valid_actions: list, device: torch_device,
                      epsilon: float) -> th.tensor:
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
        :rtype: th.tensor
        """
      sample = random.random()
      greedy = q_values is not None and sample > epsilon

      if greedy:
         actions = DRLController.greedy_policy(q_values, valid_actions)
      else:
         actions = DRLController.random_policy(valid_actions, device)

      return actions

   @staticmethod
   def _get_trr(
      network: HeteroMAGNet,
      states: list,
      next_states: list,
      non_final_next_states_indices: list,
      trr_coef: float,
   ):
      """Calculate temporal relation regularization. Attention weights from
        the last layer of the communication module of the policy network
        after processing state :math:`s` are compared against the same
        attention weights after processing state :math:`s'`. Comparison is
        done using the Kullback Leibler divergence. The larger the
        divergence, the larger the regularization factor.

        :param states: list of current states
        :type states: list
        :param next_states: list of next states
        :type next_states: list
        :param non_final_next_states_indices: indices of non final next states
        :type non_final_next_states_indices: list
        :raises NotImplementedError: if the communication layer is not a GAT layer
        :return: TRR factor
        :rtype: th.tensor
        """
      if trr_coef == 0:
         return 0

      if not isinstance(network.relational_layer, GATModule):
         raise NotImplementedError(
            "Only GAT layer type has attention weights necessary to calculate temporal relation regularization"
         )
      # turn the policy net to eval mode
      with th.no_grad():
         network.eval()

         # run the batch of current states and get attention from last layer
         valid_states = Batch.from_data_list([states[i] for i in non_final_next_states_indices])
         network(valid_states)
         edge_index = network.relational_layer.attention_indices
         att_states = network.relational_layer.last_layer_attention

         # do the same with the batch of next non-final states
         non_final_next_states_batch = Batch.from_data_list([
            next_states[i] for i in non_final_next_states_indices])
         network(non_final_next_states_batch)
         next_edge_index = network.relational_layer.attention_indices
         att_next_states = network.relational_layer.last_layer_attention

      network.train()

      # create two dense adjacency matrices to hold the organized attention weights
      # start them with zeros (absence of edges)
      final_dist1 = th.zeros(
         (
            valid_states.num_nodes,
            valid_states.num_nodes,
            network.relational_layer.n_heads,
         ),
         device=network.device,
      )
      final_dist2 = th.zeros(
         (
            non_final_next_states_batch.num_nodes,
            non_final_next_states_batch.num_nodes,
            network.relational_layer.n_heads,
         ),
         device=network.device,
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
      return trr_coef * F.kl_div(final_dist1, final_dist2, reduction="batchmean")

   def optimize(self, step_num: int, trainer: TrainingConfig):
      if not self.training_mode:
         raise RuntimeError("Optimization not allowed when not in training mode")

      loss, trr = self._optimize(trainer)

      total_loss = loss + trr

      loggers_poggers = {"losses/sum_class_loss": loss}
      if trr != 0:
         loggers_poggers["losses/trr"] = trr
         loggers_poggers["losses/total_loss"] = total_loss

      # Optimize
      self.optimizer.zero_grad()
      total_loss.backward()
      # clip gradients
      if trainer.grad_norm_clip is not None and trainer.grad_norm_clip > 0:
         th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), trainer.grad_norm_clip)

      # perform an optimization step
      self.optimizer.step()

      if step_num - self.last_grad_log > trainer.max_num_steps / 2000:
         self.last_grad_log = step_num
         for tag, value in self.policy_net.named_parameters():
            tag = tag.replace(".", "/")
            loggers_poggers[tag] = wandb.Histogram(value.data.cpu().numpy())
            if value.grad is not None:
               loggers_poggers[tag + "/grad"] = wandb.Histogram(value.grad.data.cpu().numpy())

      wandb.log(loggers_poggers, step=step_num)

   def maybe_save_checkpoint(self, step_num):
      if self.last_save is None:
         self.last_save = datetime.now()
      elif (datetime.now() - self.last_save).total_seconds() >= self.checkpoint_save_secs:
         self.save_checkpoint(step_num)

   def save_checkpoint(self, step_num):
      print("Training step: {}\n\tSaving checkpoint...".format(step_num))

      net_weights = self.get_net_state_dicts()

      common_variables = {
         "optimizer_state_dict": self.optimizer.state_dict(),
         "total_steps": step_num,
         "last_log_step": self.last_grad_log,
         "n_episodes": self.n_episodes, }

      variables = {**net_weights, **common_variables}

      th.save(variables, self.checkpoint)
      self.last_save = datetime.now()

   def get_net_state_dicts(self):
      raise NotImplementedError


class DQNController(DRLController):
   def __init__(
      self,
      checkpoint_file: str,
      action_module: TrainingConfig.ActionModuleType,
      policy: str,
      max_num_steps: int,
      batch_size: int,
      optimizer: str,
      lr: float,
      weight_decay: float,
      rmsprop_alpha: float,
      rmsprop_eps: float,
      trr_coef: float,
      checkpoint_save_secs: int,
      graph_layer_type: str,
      share_encoding: bool,
      share_comms: bool,
      share_action: bool,
      full_agent_communication: bool,
      full_receptive_field: bool,
      gat_n_heads: int,
      gat_average_last: bool,
      rgcn_n2_relations: bool,
      rgcn_num_bases: int,
      rgcn_fast: bool,
      device,
      node_types: list,
      agent_types: list,
      features_by_node_class: list,
      actions_by_node_class: list,
      training_mode: bool,
      data_parallel: bool,
      act_encoding: str,
      act_comms: str,
      act_action: str,
      use_rnn_encoding: bool,
      use_rnn_action: bool,
      gamma: float,
      eps_start,
      eps_end,
      eps_anneal_time,
      target_update: int,
      double_dqn: bool,
      mixer: str,
      prioritized_replay_eps: float = 1e-6,
      encoding_output_size=128,
      graph_module_sizes=None,
      action_hidden_size=128,
   ):
      super().__init__(
         checkpoint_file,
         action_module,
         policy,
         max_num_steps,
         batch_size,
         optimizer,
         lr,
         weight_decay,
         rmsprop_alpha,
         rmsprop_eps,
         trr_coef,
         checkpoint_save_secs,
         graph_layer_type,
         share_encoding,
         share_comms,
         share_action,
         full_agent_communication,
         full_receptive_field,
         gat_n_heads,
         gat_average_last,
         rgcn_n2_relations,
         rgcn_num_bases,
         rgcn_fast,
         device,
         node_types,
         agent_types,
         features_by_node_class,
         actions_by_node_class,
         training_mode,
         data_parallel,
         act_encoding,
         act_comms,
         act_action,
         use_rnn_encoding,
         use_rnn_action,
         mixer,
         encoding_output_size,
         graph_module_sizes,
         action_hidden_size,
      )
      self.gamma = gamma
      self.target_update = target_update
      self.double_dqn = double_dqn
      self.prioritized_replay_eps = prioritized_replay_eps
      self.eps_scheduler = DecayThenFlatSchedule(eps_start, eps_end, eps_anneal_time, "linear")

      self.target_net = HeteroMAGNet(
         self.action_layer_type,
         share_encoding,
         share_comms,
         share_action,
         self.num_unit_types,
         agent_types,
         node_types,
         features_by_node_class,
         encoding_output_size,
         self.graph_module_sizes,
         self.action_hidden_size,
         actions_by_node_class,
         act_encoding,
         act_comms,
         act_action,
         use_rnn_encoding,
         use_rnn_action,
         self.device,
         self.graph_layer_type,
         self.full_receptive_field,
         self.gat_n_heads,
         self.gat_average_last,
         self.rgcn_n2_relations,
         self.rgcn_num_bases,
         self.rgcn_fast,
         mixer,
      )

      self.target_net = DRLController.maybe_parallel_net(self.target_net,
                                                         self.device,
                                                         data_parallel)

      nn_trainable_parameters = sum(p.numel() for p in self.policy_net.parameters()
                                    if p.requires_grad)
      print("Neural network has {} trainable parameters".format(nn_trainable_parameters))

      if isfile(self.checkpoint):
         print("Loading from checkpoint...")
         checkpoint = th.load(self.checkpoint)
         self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
         self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
         self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
         # self.total_steps = checkpoint['total_steps']
         self.last_grad_log = checkpoint["last_log_step"]
         self.n_episodes = checkpoint["n_episodes"]
      elif not self.training_mode:
         raise FileNotFoundError(
            "Checkpoint file {} does not exist. It is necessary in order to play a game.".format(
               self.checkpoint))
      else:
         self.target_net.load_state_dict(self.policy_net.state_dict())

      # wandb.watch(self.policy_net, log='all', log_freq=250)

      self.policy_net.train(self.training_mode)
      self.target_net.eval()

   def update_target_net(self):
      self.target_net.load_state_dict(self.policy_net.state_dict())

   def get_net_state_dicts(self):
      return {
         "policy_net_state_dict": self.policy_net.state_dict(),
         "target_net_state_dict": self.target_net.state_dict(), }

   def act(
      self,
      output_dict,
      valid_actions: list,
      step_num: int,
      epsilon: int = 1,
      evaluation=False,
   ) -> th.tensor:
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
        :rtype: th.tensor
        """
      BaseController._action_lists_to_tensors(valid_actions)

      if evaluation or self.policy == DRLController.Policy.GREEDY:
         actions = self.greedy_policy(output_dict, valid_actions)
      elif self.policy == DRLController.Policy.RANDOM:
         actions = self.random_policy(valid_actions, self.device)
      # elif self.policy == MultiAgentDRLAlgorithm.Policy.BOLTZMANN:
      #    actions = self.boltzmann_policy(output_dict, valid_actions, self.device)
      elif self.policy == DRLController.Policy.EGREEDY:
         actions = self.egreedy_policy(output_dict, valid_actions, self.device, epsilon)
      elif self.policy == DRLController.Policy.EGREEDY_DECAY:
         epsilon = self.eps_scheduler.eval(step_num)
         actions = self.egreedy_policy(output_dict, valid_actions, self.device, epsilon)
         if step_num % 250 == 0:
            wandb.log({"epsilon": epsilon}, step=step_num)
      else:
         raise NotImplementedError("this policy doesn't exist!")

      return actions

   @staticmethod
   def normalize_batch_length(eps, device):
      max_size = max([len(ep) for ep in eps])
      # create dummy data for the padding states
      # TODO the environment or the Runner could provide a dummy state
      sample_state = eps[0][0].state
      dummy_x = th.zeros_like(sample_state.x)
      dummy_edge_index = th.tensor([[], []], dtype=th.long)

      dummy_state = Data(x=dummy_x, edge_index=dummy_edge_index).to(device=device)
      # dummy_actions = th.zeros(sample_state.x.size(0), dtype=th.long)

      # NOTE This has been hardcoded for SMAC
      dummy_av_actions = (th.tensor([True] + [False] *
                                    (eps[0][0].av_actions[0].size(0) - 1)).unsqueeze(1).repeat(
                                       1, sample_state.x.size(0)).T)

      action_batch = []
      reward_batch = []
      done_batch = []
      av_actions_batch = []
      state_batch = []
      dummy = th.ones((len(eps), max_size), dtype=th.bool, device=device)

      for i, episode in enumerate(eps):
         batch = Transition(*zip(*episode))

         dummy[i, :len(episode)] = 0

         states_in_device = [s.clone().to(device)
                             for s in batch.state] + [dummy_state] * (max_size - len(batch.state))

         action = th.stack(batch.action).long()
         action = th.cat((
            action,
            th.zeros((max_size - action.size(0), action.size(1)), dtype=th.long),
         ))
         action_batch.append(action.unsqueeze(2))
         # action_batch.append(action.unsqueeze())

         reward = th.tensor(list(batch.reward) + [0] * (max_size - len(batch.reward)),
                            device=device)
         reward_batch.append(reward)

         done = th.tensor(list(batch.done) + [1] * (max_size - len(batch.done)))
         done_batch.append(done.flatten())

         av_actions = th.stack([th.stack(av) for av in batch.av_actions])
         dummy_av_actions_for_state = dummy_av_actions.unsqueeze(0).repeat(
            max_size - av_actions.size(0), 1, 1)

         av_actions_batch.append(th.cat((av_actions, dummy_av_actions_for_state), dim=0))

         state_batch.append(states_in_device)

      action_batch = th.stack(action_batch).to(device)
      reward_batch = th.stack(reward_batch).to(device)
      done_batch = th.stack(done_batch).to(device)
      av_actions_batch = th.stack(av_actions_batch).to(device)

      return (
         state_batch,
         action_batch,
         reward_batch,
         done_batch,
         av_actions_batch,
         dummy,
      )

   def optimize(self, step_num: int, trainer: TrainingConfig, memory):
      assert memory.can_sample(
          trainer.batch_size
      ), "Replay memory hasn't reached the minimum size for an optimization step"

      self.policy_net.train()

      episodes = (memory.sample(trainer.batch_size) if not memory.is_prioritized else memory.sample(
         trainer.batch_size, trainer.replay_buffer_beta))

      if memory.is_prioritized:
         episodes, _, batch_idxes = episodes

      # state_batch, action_batch, reward_batch, done_batch, av_actions_batch = self.normalize_to_smallest(episodes, self.device)
      (
         state_batch,
         action_batch,
         reward_batch,
         done_batch,
         av_actions_batch,
         dummy,
      ) = self.normalize_batch_length(episodes, self.device)

      # compute current and target Q vals
      # episodes x time steps x agents x actions
      # NOTE we take advantage of the fact SMAC agents have
      # same number of actions to preallocate this tensor
      current_q_vals = th.empty(
         len(state_batch),
         len(state_batch[0]),
         self.n_agents,
         self.n_actions_agents[0],
         device=self.device,
      )
      target_q_vals = th.empty(
         len(state_batch),
         len(state_batch[0]),
         self.n_agents,
         self.n_actions_agents[0],
         device=self.device,
      )

      if self.policy_net.encoding_layer.use_rnn:
         self.policy_net.encoding_layer.init_hidden(len(state_batch))
         self.target_net.encoding_layer.init_hidden(len(state_batch))
      if self.policy_net.action_layer.use_rnn:
         self.policy_net.action_layer.init_hidden(len(state_batch))
         self.target_net.action_layer.init_hidden(len(state_batch))

      for t in range(len(state_batch[0])):
         time_step = [episode[t].to(self.device) for episode in state_batch]
         graph_state_batch = Batch.from_data_list(time_step).to(device=self.device)
         current_q_vals[:, t, :, :] = self.policy_net(graph_state_batch)
         target_q_vals[:, t, :, :] = self.target_net(graph_state_batch)

      del state_batch

      # ignore first states in episodes, mask unavailable actions
      # NOTE I tried masking with float("-inf") but CUDA tended to run out of memory?
      target_q_vals = target_q_vals[:, 1:]
      target_q_vals[~av_actions_batch[:, 1:]] = float("-inf")

      # Compute max q(s',a) for all non-final next states
      if not self.double_dqn:
         max_next_qvals = target_q_vals.max(3)[0]
      else:
         # at first I used the network again: next_q_vals = self.policy_net(non_final_next_states)
         # but then I saw I could just use the current q_vals, but for the next states
         live_next_q_vals = current_q_vals.clone().detach()[:, 1:]
         live_next_q_vals[~av_actions_batch[:, 1:]] = float("-inf")
         max_next_actions = live_next_q_vals.max(3, keepdim=True)[1]
         max_next_qvals = target_q_vals.gather(3, max_next_actions).squeeze()
         del live_next_q_vals, max_next_actions

      del target_q_vals, av_actions_batch

      # select the Q vals of actions taken by the agents in each state
      chosen_actions_qvals = th.gather(current_q_vals, 3, action_batch).squeeze()
      del current_q_vals

      # apply mixer
      if self.policy_net.mixer is not None:
         chosen_actions_qvals = self.policy_net.mixer(chosen_actions_qvals)  # [:, :-1]
         max_next_qvals = self.target_net.mixer(max_next_qvals)  # [:, 1:]

      # Compute expected Q values
      r = reward_batch[:, 1:].unsqueeze(2).expand_as(max_next_qvals)
      gamma_and_mask = ((self.gamma *
                         (1 - done_batch[:, :-1])).unsqueeze(2).expand_as(max_next_qvals))
      expected_state_action_values_by_class = r + gamma_and_mask * max_next_qvals
      del max_next_qvals, r, gamma_and_mask

      # TD error
      td_errors = (chosen_actions_qvals[:, :-1] - expected_state_action_values_by_class.detach())

      mask = (~dummy[:, :-1]).float()
      mask[:, 1:] = mask[:, 1:] * (1 - done_batch[:, :-2])

      # TODO I think this is always true
      if td_errors.ndim > mask.ndim:
         mask = mask.unsqueeze(-1)
      mask = mask.expand_as(td_errors)

      # zero-out the targets that came from padded data
      masked_td_error = td_errors * mask

      if memory.is_prioritized:
         masked_td_error = td_errors.abs() * mask

         if trainer.episode_priority == "mean":
            # mean TD-error
            priorities = masked_td_error.sum((1, 2)) / mask.sum((1, 2))
         elif trainer.episode_priority == "max":
            # max TD-error in the episode
            # I use the hack below because there is no max operation along multiple dimensions
            priorities = masked_td_error
            while priorities.ndim > 1:
               priorities = priorities.max(1)[0].squeeze()
         elif trainer.episode_priority == "median":
            # TODO need to find a way to get the median of a masked tensor along an axis
            # priorities = masked_td_error[mask].median(1)
            raise NotImplementedError
         else:
            raise NotImplementedError

         memory.update_priorities(batch_idxes.detach().tolist(), priorities.detach().tolist())

      # Normal L2 loss, take mean over actual data
      total_loss = (masked_td_error**2).sum() / mask.sum()

      # loss
      # total_loss = td_errors.pow(2).mean()
      # loss = td_errors.pow(2).mean()

      # calculate temporal relation regularization from Jiang 2020
      # trr = self._get_trr(self.policy_net,
      #                     state_batch[:-1],
      #                     state_batch[1:],
      #                     non_final_next_states_indices,
      #                     self.trr_coef)

      # total_loss = loss + trr

      # if trr != 0:
      #    loggers_poggers['losses/trr'] = trr

      # Optimize
      # self.optimizer.zero_grad()
      for p in self.policy_net.parameters():
         p.grad = None

      total_loss.backward()
      # clip gradients
      if trainer.grad_norm_clip is not None and trainer.grad_norm_clip > 0:
         th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), trainer.grad_norm_clip)

      # perform an optimization step
      self.optimizer.step()

      loggers_poggers = {"losses/loss": total_loss.detach().item()}
      if step_num - self.last_grad_log > trainer.max_num_steps / 2000:
         self.last_grad_log = step_num
         # for tag, value in self.policy_net.named_parameters():
         #    tag = tag.replace('.', '/')
         #    loggers_poggers[tag] = wandb.Histogram(value.data.cpu().numpy())
         #    if value.grad is not None:
         #       loggers_poggers[tag + '/grad'] = wandb.Histogram(value.grad.data.cpu().numpy())
      wandb.log(loggers_poggers, step=step_num)
