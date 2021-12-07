import argparse
import platform
import warnings
from collections import OrderedDict
from enum import Enum
from os import path

import GPUtil
import torch as th

from ..nn.activation import activation_functions


class TrainingConfig:
   """Object that integrates all parameters related to a neural network training.
    Responsible for collecting and processing command-line arguments, loading
    checkpoints, storing, maintaining and creating backups of the replay buffer
    and logging training progress in a per episode basis.
    """
   class GameType(Enum):
      """Enumerates the types of game environments supported by the trainer"""

      SMAC = "SMAC"
      """A game in the StarCraft Multi-Agent Challenge"""

   class ActionModuleType(Enum):
      DQN = "Deep Q-Network"
      DDQN = "Dueling Deep Q-Network"
      RANDOM = "Random"

   def __init__(self):
      self.action_hidden_size = None
      self.action_module = None
      self.batch_size = None
      self.checkpoint_file = None
      self.checkpoint_save_secs = None
      self.comms_sizes = None
      self.device = None
      self.double_dqn = None
      self.encoding_hidden_size = None
      self.episode_num = None
      self.episode_priority = None
      self.eps_anneal_time = None
      self.eps_end = None
      self.eps_start = None
      self.eval_episodes = None
      self.eval_interval = None
      self.full_agent_communication = None
      self.full_receptive_field = None
      self.game = None
      self.game_name = None
      self.gamma = None
      self.gat_average_last = None
      self.gat_n_heads = None
      self.grad_norm_clip = None
      self.graph_layer_type = None
      self.log_dir = None
      self.lr = None
      self.max_num_episodes = None
      self.max_num_steps = None
      self.max_steps_episode = None
      self.max_trajectory_length = None
      self.mixer = None
      self.new_run = None
      self.rmsprop_alpha = None
      self.rmsprop_eps = None
      self.pbar = None
      self.policy = None
      self.render = None
      self.replay_buffer_alpha = None
      self.replay_buffer_beta = None
      self.replay_buffer_file = None
      self.replay_buffer_save_interval = None
      self.replay_buffer_size = None
      self.resume_run = None
      self.rgcn_fast = None
      self.rgcn_n2_relations = None
      self.rgcn_num_bases = None
      self.run_prefix = None
      self.save_replays = None
      self.share_action = None
      self.share_comms = None
      self.share_encoding = None
      self.sparse_rewards = None
      self.target_update = None
      self.training_mode = None
      self.trr_coef = None
      self.v2_state = None
      self.weight_decay = None

      self.actor_coef = None
      self.critic_coef = None
      self.entropy_coef = None
      self.entropies: th.tensor
      self.bootstrap_values: th.tensor

   def initialize(self):
      parser = argparse.ArgumentParser(description="Parse.")
      parser.add_argument(
         "game",
         default="SMAC",
         choices=["SMAC"],
         type=str,
         help="which environment to use",
      )
      parser.add_argument("action_module",
                          default="DQN",
                          choices=["DQN", "DDQN", "RANDOM"],
                          type=str)
      parser.add_argument("game_name", type=str, help="SMAC map name")
      parser.add_argument("run_name", type=str, help="run name for logging and checkpoint purposes")
      parser.add_argument("--run_prefix", default="default", type=str)
      parser.add_argument(
         "--render",
         type=int,
         choices=list(range(4)),
         default=0,
         help=
         "Real-time rendering. 0=no; 1=eval; 2=training; 3=everything. Rendering is slower than no rendering, I suggest not using it for training.",
      )
      parser.add_argument("--batch_size", default=32, type=int, help="batch size")
      parser.add_argument(
         "--checkpoint_save_secs",
         default=180,
         type=int,
         help="seconds between nn checkpoint saves",
      )
      parser.add_argument(
         "--device",
         default="smart",
         type=str,
         help=
         'torch device ("cpu", "cuda", "cuda:0" etc.). "smart" selects the CPU if no GPU available, or the only GPU, or the GPU with most memory available, or errors',
      )
      parser.add_argument(
         "--dry_run",
         dest="dry_run",
         action="store_true",
         default=False,
         help="dry run to instantiate environment, network, get its size",
      )
      parser.add_argument("--eps_end", default=0.1, type=float, help="final number of epsilon")
      parser.add_argument("--eps_start", default=1, type=float, help="initial number of epsilon")
      parser.add_argument(
         "--eps_anneal_time",
         default=50000,
         type=float,
         help="step at which epsilon will be equal to eps_end",
      )
      parser.add_argument(
         "--eval",
         dest="training_mode",
         action="store_false",
         default=True,
         help="don't train a neural network, load one from a file for evaluation",
      )
      parser.add_argument(
         "--eval_episodes",
         default=32,
         type=int,
         help="number of episodes in each evaluation cycle",
      )
      parser.add_argument(
         "--eval_interval",
         default=10000,
         type=int,
         help="number of steps between evaluation cycles",
      )
      parser.add_argument(
         "--data_parallel",
         dest="data_parallel",
         action="store_true",
         default=False,
         help="use multiple GPUs if available (not working currently)",
      )
      parser.add_argument(
         "--full_agent_communication",
         dest="full_agent_communication",
         action="store_true",
         default=False,
         help="generate missing edges between all agents",
      )
      parser.add_argument(
         "--full_receptive_field",
         dest="full_receptive_field",
         action="store_true",
         default=False,
         help="the graph module outputs the output of all layers, instead of only the last one",
      )
      parser.add_argument("--gamma", default=0.99, type=float, help="RL discount factor")
      parser.add_argument(
         "--gat_average_last",
         dest="gat_average_last",
         action="store_true",
         default=False,
         help="average the last GAT layer, instead of concatenating",
      )
      parser.add_argument(
         "--gat_n_heads",
         default=3,
         type=int,
         help="number of attention heads in the GAT module",
      )
      parser.add_argument(
         "--grad_norm_clip",
         default=10,
         type=float,
         help="global gradient norm clipping",
      )
      parser.add_argument(
         "--episode_priority",
         default=None,
         type=str,
         choices=["mean", "max", "median"],
      )
      parser.add_argument("--graph_layer_type",
                          default=None,
                          type=str,
                          choices=["GCN", "GAT", "RGCN"])
      parser.add_argument(
         "--lr",
         default=0.0005,
         type=float,
         help="learning rate of the neural network",
      )
      parser.add_argument(
         "--max_num_episodes",
         default=None,
         type=int,
         help="maximum number of episodes",
      )
      parser.add_argument(
         "--max_num_steps",
         default=1000000,
         type=int,
         help="maximum number of training steps",
      )
      parser.add_argument(
         "--max_steps_episode",
         default=None,
         type=int,
         help="maximum number of steps inside an episode, before resetting the environment",
      )
      parser.add_argument(
         "--max_trajectory_length",
         default=None,
         type=int,
         help="maximum length of trajectories in on-policy methods",
      )
      parser.add_argument("--mixer", default=None, type=str, help="whether use a mixer (vdn)")
      parser.add_argument(
         "--new_run",
         dest="new_run",
         action="store_true",
         default=False,
         help="start a new run, otherwise continue last one with same hyperparameters",
      )
      parser.add_argument(
         "--optimizer",
         default="rmsprop",
         type=str,
         choices=["rmsprop", "adam"],
         help="neural network optimizer",
      )
      parser.add_argument(
         "--replay_buffer_alpha",
         default=0.6,
         type=float,
         help="value for the alpha param of the prioritized replay buffer",
      )
      parser.add_argument(
         "--replay_buffer_beta",
         default=0.4,
         type=float,
         help="value for the beta param of the prioritized replay buffer",
      )
      parser.add_argument(
         "--replay_buffer_size",
         default=5000,
         type=int,
         help="maximum number of transitions held in the replay buffer",
      )
      parser.add_argument(
         "--resume_run",
         action="store_true",
         default=False,
         help="whether to resume from the training step where left off",
      )
      parser.add_argument(
         "--rgcn_num_bases",
         default=1,
         type=int,
         help="number of basis matrices for RGCN layers",
      )
      parser.add_argument(
         "--rgcn_fast",
         dest="rgcn_fast",
         action="store_true",
         default=False,
         help="use FastRGCN instead of RGCN",
      )
      parser.add_argument("--rmsprop_alpha", default=0.99, type=float, help="RMSprop alpha")
      parser.add_argument("--rmsprop_eps", default=0.00001, type=float, help="RMSprop eps")
      parser.add_argument(
         "--sparse_rewards",
         dest="sparse_rewards",
         action="store_true",
         default=False,
         help="use sparse rewards (1, 0, -1 at end of episode), instead of dense",
      )
      parser.add_argument(
         "--save_replays",
         dest="save_replays",
         action="store_true",
         default=False,
         help=
         "save replays from evaluation episodes, ((max_steps / eval_interval) * eval_episodes) SC2Replay files will be created",
      )
      parser.add_argument(
         "--share_encoding",
         dest="share_encoding",
         action="store_true",
         default=False,
         help="share the same network in the encoding module for all agent classes",
      )
      parser.add_argument(
         "--share_comms",
         dest="share_comms",
         action="store_true",
         default=False,
         help=
         "share the same parameters in the communication module for all agent classes (in the case of RGCN only)",
      )
      parser.add_argument(
         "--share_action",
         dest="share_action",
         action="store_true",
         default=False,
         help="share the same network in the action module for all agent classes",
      )
      parser.add_argument(
         "--trr_coef",
         default=0,
         type=float,
         help="coefficient for temporal relation regularization",
      )
      parser.add_argument("--weight_decay", default=0, type=float, help="L2 regularization factor")
      parser.add_argument(
         "--encoding_hidden",
         default=128,
         type=int,
         help="number of neurons in the FC/LSTM layer of the encoding nets",
      )
      parser.add_argument(
         "--comms_sizes",
         default="128,128",
         type=str,
         help="number of neurons in each layer of the communication module",
      )
      parser.add_argument(
         "--action_hidden",
         default=128,
         type=int,
         help="number of neurons in hidden layer of the action nets",
      )
      parser.add_argument(
         "--v2_state",
         dest="v2_state",
         action="store_true",
         default=False,
         help="use version 2 of the graph state representation",
      )
      parser.add_argument(
         "--use_rnn_encoding",
         dest="use_rnn_encoding",
         action="store_true",
         default=False,
         help="",
      )
      parser.add_argument(
         "--use_rnn_action",
         dest="use_rnn_action",
         action="store_true",
         default=False,
         help="",
      )

      parser.add_argument(
         "--act_encoding",
         default="sigmoid",
         type=str,
         choices=activation_functions.keys(),
         help="activation function",
      )
      parser.add_argument(
         "--act_comms",
         default="sigmoid",
         type=str,
         choices=activation_functions.keys(),
         help="",
      )
      parser.add_argument(
         "--act_action",
         default="sigmoid",
         type=str,
         choices=activation_functions.keys(),
         help="",
      )
      # off-policy methods
      parser.add_argument(
         "--policy",
         default="egreedy_decay",
         type=str,
         choices=["random", "boltzmann", "greedy", "egreedy", "egreedy_decay"],
         help="(OFF-POLICY) policy type",
      )
      parser.add_argument(
         "--target_update",
         default=150,
         type=int,
         help="(OFF-POLICY) number of steps between updating the target network",
      )
      parser.add_argument(
         "--double_dqn",
         dest="double_dqn",
         action="store_true",
         default=False,
         help="use Double DQN method",
      )

      # actor-critic
      parser.add_argument(
         "--actor_coef",
         default=1,
         type=float,
         help="(AC) coefficient for actor loss in actor-critic methods",
      )
      parser.add_argument(
         "--critic_coef",
         default=0.5,
         type=float,
         help="(AC) coefficient for critic loss in actor-critic methods",
      )
      parser.add_argument(
         "--entropy_coef",
         default=0.01,
         type=float,
         help="(AC) coefficient for entropy loss in actor-critic methods",
      )
      parser.add_argument(
         "--rgcn_n1_relations",
         dest="rgcn_n2_relations",
         action="store_false",
         default=True,
         help="use one relation per agent-type in RGCN",
      )

      args = parser.parse_args()
      print(args)

      if not args.training_mode and args.new_run:
         raise ValueError("Can't start a new run in eval mode, need checkpoint file to start from")

      if args.trr_coef != 0 and args.graph_layer_type.upper() == "RGCN":
         args.trr_coef = 0
         warnings.warn(
            "TRR coef is nonzero but comms layer has no attention weights. TRR will be ignored.")

      assert args.replay_buffer_size >= args.batch_size

      # required args
      self.game = TrainingConfig.GameType[args.game.upper()]
      self.game_name = args.game_name
      self.action_module = TrainingConfig.ActionModuleType[args.action_module.upper()]

      # common args
      self.act_action = args.act_action
      self.act_comms = args.act_comms
      self.act_encoding = args.act_encoding
      self.batch_size = args.batch_size
      self.checkpoint_save_secs = args.checkpoint_save_secs
      self.data_parallel = args.data_parallel
      self.dry_run = args.dry_run
      self.episode_priority = args.episode_priority
      self.eval_episodes = args.eval_episodes
      self.eval_interval = args.eval_interval
      self.full_agent_communication = args.full_agent_communication
      self.full_receptive_field = args.full_receptive_field
      self.gat_average_last = args.gat_average_last
      self.gat_n_heads = args.gat_n_heads
      self.grad_norm_clip = args.grad_norm_clip
      self.graph_layer_type = (args.graph_layer_type.upper()
                               if args.graph_layer_type is not None else None)
      self.lr = args.lr
      self.max_num_episodes = args.max_num_episodes
      self.max_num_steps = args.max_num_steps
      self.max_steps_episode = args.max_steps_episode
      self.mixer = args.mixer
      self.new_run = args.new_run
      self.optimizer = args.optimizer
      self.rmsprop_alpha = args.rmsprop_alpha
      self.rmsprop_eps = args.rmsprop_eps
      self.render_eval = args.render in [1, 3]
      self.render_train = args.render >= 2
      self.resume_run = args.resume_run
      self.rgcn_fast = args.rgcn_fast
      self.rgcn_n2_relations = args.rgcn_n2_relations
      self.rgcn_num_bases = args.rgcn_num_bases
      self.run_prefix = args.run_prefix
      self.save_replays = args.save_replays
      self.sparse_rewards = args.sparse_rewards
      self.training_mode = args.training_mode
      self.trr_coef = args.trr_coef
      self.use_rnn_action = args.use_rnn_action
      self.use_rnn_encoding = args.use_rnn_encoding
      self.v2_state = args.v2_state
      self.weight_decay = args.weight_decay
      self.share_action = args.share_action
      self.share_comms = args.share_comms
      self.share_encoding = args.share_encoding

      self.step_num = 0

      # network sizes
      self.encoding_hidden_size = args.encoding_hidden
      self.comms_sizes = list(map(int, args.comms_sizes.split(",")))
      self.action_hidden_size = args.action_hidden

      # off-policy args
      self.double_dqn = args.double_dqn
      self.eps_anneal_time = args.eps_anneal_time
      self.eps_end = args.eps_end
      self.eps_start = args.eps_start
      self.gamma = args.gamma
      self.policy = args.policy.upper()
      self.replay_buffer_alpha = args.replay_buffer_alpha
      self.replay_buffer_beta = args.replay_buffer_beta
      self.replay_buffer_size = args.replay_buffer_size
      self.target_update = args.target_update

      # on-policy args
      self.actor_coef = args.actor_coef
      self.critic_coef = args.critic_coef
      self.entropy_coef = args.entropy_coef

      dev = args.device
      if dev == "smart":
         gpus = GPUtil.getGPUs()
         if len(gpus) == 0:
            dev = "cpu"
         elif len(gpus) == 1:
            dev = "cuda"
         else:
            available_gpus = GPUtil.getAvailable(order="memory")
            if len(available_gpus) == 0:
               raise RuntimeError("No GPUs available to start training")
            dev = "cuda:" + str(available_gpus[0])

      self.device = th.device(dev)
      self.run_name, self.log_dir = self.generate_run_name(args.run_name)

   def get_loggable_args(self):
      d = OrderedDict()
      d["MAIN"] = [
         ("run_prefix", self.run_prefix),
         ("hostname", platform.node()),
         ("game", self.game.name),
         ("game_name", self.game_name),
         ("action_module", self.action_module.name),
         ("graph_version", "v2" if self.v2_state else "v1"),
         #  ('max_num_steps', self.max_num_steps),
         #  ('max_num_episodes',
         #   0 if self.max_num_episodes is None else self.max_num_episodes),
         #  ('max_steps_episode',
         #   0 if self.max_steps_episode is None else self.max_steps_episode)
      ]

      if self.action_module != TrainingConfig.ActionModuleType.RANDOM:
         d["NETWORK"] = [
            ("batch_size", self.batch_size),
            ("share_action", self.share_action),
            ("share_comms", self.share_comms),
            ("share_encoding", self.share_encoding),
            ("grad_norm_clip", self.grad_norm_clip),
            ("device", self.device.type),
            (
               "full_agent_communication",
               "fullcomms" if self.full_agent_communication else "nofullcomms",
            ),
            (
               "full_receptive_field",
               "fullrcp" if self.full_receptive_field else "nofullrcp",
            ),
            ("gat_average_last", "avg" if self.gat_average_last else "concat"),
            ("gat_n_heads", self.gat_n_heads),
            ("graph_layer_type", self.graph_layer_type),
            ("lr", self.lr),
            ("rgcn_n2_relations", "n2rels" if self.rgcn_n2_relations else "n1rels"),
            ("rgcn_num_bases", self.rgcn_num_bases),
            ("rgcn_fast", self.rgcn_fast),
            ("sparse_rewards", "sparse" if self.sparse_rewards else "dense"),
            ("trr_coef", self.trr_coef),
            ("weight_decay", self.weight_decay), ]

         if self.action_module in [
               TrainingConfig.ActionModuleType.DQN,
               TrainingConfig.ActionModuleType.DDQN, ]:
            d["OFF-POLICY"] = [
               ("eps_end", self.eps_end),
               ("eps_start", self.eps_start),
               ("gamma", self.gamma),
               ("replay_buffer_alpha", self.replay_buffer_alpha),
               ("replay_buffer_beta", self.replay_buffer_beta),
               ("target_update", self.target_update),
               ("double_dqn", self.double_dqn),
               ("policy", self.policy),
               ("replay_buffer_size", self.replay_buffer_size), ]

      return d

   def generate_run_name(self, run_name):
      userhome = path.expanduser("~")
      log_dir = path.join(
         userhome,
         ".hcanet",
         self.game.value,
         self.game_name,
         self.run_prefix,
         run_name,
      )

      if not self.new_run or not path.isdir(log_dir):
         return run_name, log_dir

      run_name += "_"
      log_dir += "_"
      run_num = 1
      run_name_0 = run_name + str(run_num)
      log_dir_0 = log_dir + str(run_num)
      while path.isdir(log_dir_0):
         run_num += 1
         run_name_1 = run_name + str(run_num)
         log_dir_1 = log_dir + str(run_num)

         if not path.isdir(log_dir_1):
            return ((run_name_1, log_dir_1) if self.new_run else (run_name_0, log_dir_0))

         log_dir_0 = log_dir_1
         run_name_0 = run_name_1

      return run_name_0, log_dir_0
