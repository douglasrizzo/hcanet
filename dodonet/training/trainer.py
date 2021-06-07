import argparse
from datetime import datetime
from enum import IntEnum
from os.path import expanduser, isfile
from os.path import join as path_join

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dodonet.rl.replay import PrioritizedReplayBuffer


class Trainer:
    """Object that integrates all parameters related to a neural network training.
    Responsible for collecting and processing command-line arguments, loading
    checkpoints, storing, maintaining and creating backups of the replay buffer
    and logging training progress in a per episode basis.

    :param game_type: What environment will be used during the training process
    :type game_type: GameType
    """

    class GameType(IntEnum):
        """Enumerates the types of game environments supported by the trainer"""
        SMAC = 0
        """A game in the StarCraft Multi-Agent Challenge"""
        POMMERMAN = 1
        """A Pommerman game"""

    def __init__(self, game_type: GameType):
        self.game_type = game_type

        self.batch_size = None
        self.checkpoint_file = None
        self.checkpoint_name = None
        self.checkpoint_save_secs = None
        self.coalesce_state_obs = None
        self.cuda_enable = None
        self.device = None
        self.episode_num = None
        self.eps_end = None
        self.eps_start = None
        self.full_agent_communication = None
        self.full_receptive_field = None
        self.game_name = None
        self.gamma = None
        self.gat_average_last = None
        self.gat_n_heads = None
        self.graph_layer_type = None
        self.log_dir = None
        self.lr = None
        self.max_steps_episode = None
        self.max_num_steps = None
        self.max_num_episodes = None
        self.max_trajectory_length = None
        self.memory = None
        self.min_memory_size = None
        self.pbar = None
        self.policy = None
        self.replay_buffer_alpha = None
        self.replay_buffer_beta = None
        self.replay_buffer_file = None
        self.replay_buffer_size = None
        self.replay_buffer_save_interval = None
        self.rgcn_n2_relations = None
        self.rgcn_num_bases = None
        self.sparse_rewards = None
        self.step_num = None
        self.summary_writer: SummaryWriter = None
        self.target_update = None
        self.training_datetime = None
        self.trr_coef = None
        self.weight_decay = None

        self.entropies: torch.Tensor = None
        self.on_policy_trajectories: list = []

    def initialize(self):
        parser = argparse.ArgumentParser(description='Parse.')
        parser.add_argument(
            'game_name',
            default='2s3z' if self.game_type == Trainer.GameType.SMAC else 'PommeRadio-v2',
            type=str,
            help='SMAC map name/Pommerman game name'
        )
        parser.add_argument('checkpoint_name', default=None, type=str)
        parser.add_argument('--batch_size', default=32, type=int, help='batch size')
        parser.add_argument(
            '--checkpoint_save_secs',
            default=180,
            type=int,
            help='seconds between nn checkpoint saves'
        )
        parser.add_argument(
            '--coalesce_state_obs',
            dest='coalesce_state_obs',
            action='store_true',
            default=False,
            help='concat the original agent state to its observation, provenient from the communication layer'
        )
        parser.add_argument(
            '--cuda',
            dest='cuda_enable',
            action='store_true',
            default=False,
            help='use CUDA devices'
        )
        parser.add_argument('--eps_end', default=0.05, type=float, help='final number of epsilon')
        parser.add_argument('--eps_start', default=1, type=float, help='initial number of epsilon')
        parser.add_argument(
            '--full_agent_communication',
            dest='full_agent_communication',
            action='store_true',
            default=False,
            help='generate missing edges between all agents'
        )
        parser.add_argument(
            '--full_receptive_field',
            dest='full_receptive_field',
            action='store_true',
            default=False,
            help='the graph module outputs the output of all layers, instead of only the last one'
        )
        parser.add_argument('--gamma', default=0.99, type=float, help='RL discount factor')
        parser.add_argument(
            '--gat_average_last',
            dest='gat_average_last',
            action='store_true',
            default=False,
            help='average the last GAT layer, instead of concatenating'
        )
        parser.add_argument(
            '--gat_n_heads',
            default=3,
            type=int,
            help='number of attention heads in the GAT module'
        )
        parser.add_argument('--graph_layer_type', default='GAT', type=str, help='GCN, GAT or RGCN')
        parser.add_argument(
            '--lr', default=0.00025, type=float, help='learning rate of the neural network'
        )
        parser.add_argument(
            '--max_num_episodes', default=None, type=int, help='maximum number of episodes'
        )
        parser.add_argument(
            '--max_num_steps', default=1000000, type=int, help='maximum number of training steps'
        )
        parser.add_argument(
            '--max_steps_episode',
            default=None,
            type=int,
            help='maximum number of steps inside an episode, before resetting the environment'
        )
        parser.add_argument(
            '--max_trajectory_length',
            default=None,
            type=int,
            help='maximum length of trajectories in on-policy methods'
        )
        parser.add_argument(
            '--policy',
            default='EGREEDY_DECAY',
            type=str,
            help='policy type (RANDOM, BOLTZMANN, GREEDY, EGREEDY, EGREEDY_DECAY)'
        )
        parser.add_argument(
            '--replay_buffer_alpha',
            default=.6,
            type=float,
            help='value for the alpha param of the prioritized replay buffer'
        )
        parser.add_argument(
            '--replay_buffer_beta',
            default=.4,
            type=float,
            help='value for the beta param of the prioritized replay buffer'
        )
        parser.add_argument(
            '--replay_buffer_size',
            default=5000,
            type=int,
            help='maximum number of transitions held in the replay buffer'
        )
        parser.add_argument(
            '--rgcn_num_bases',
            default=1,
            type=int,
            help='number of basis matrices for RGCN layers'
        )
        parser.add_argument(
            '--sparse_rewards',
            dest='sparse_rewards',
            action='store_true',
            default=False,
            help='use sparse rewards (1, 0, -1 at end of episode), instead of dense'
        )
        parser.add_argument(
            '--target_update',
            default=150,
            type=int,
            help='number of steps between updating the target network'
        )
        parser.add_argument(
            '--trr_coef',
            default=0,
            type=float,
            help='coefficient for temporal relation regularization'
        )
        parser.add_argument(
            '--weight_decay', default=1e-5, type=float, help='L2 regularization factor'
        )

        feature_parser = parser.add_mutually_exclusive_group(required=False)
        feature_parser.add_argument(
            '--rgcn_n2_relations',
            dest='rgcn_n2_relations',
            action='store_true',
            help='(DEFAULT) use one relation for each pair of agent-type/node-type in RGCN'
        )
        feature_parser.add_argument(
            '--rgcn_n1_relations',
            dest='rgcn_n2_relations',
            action='store_false',
            help='use one relation per agent-type in RGCN'
        )
        parser.set_defaults(rgcn_n2_relations=True)

        args = parser.parse_args()
        print(args)

        assert args.replay_buffer_size >= args.batch_size

        self.batch_size = args.batch_size
        self.checkpoint_name = args.checkpoint_name
        self.checkpoint_save_secs = args.checkpoint_save_secs
        self.coalesce_state_obs = args.coalesce_state_obs
        self.cuda_enable = args.cuda_enable
        self.eps_end = args.eps_end
        self.eps_start = args.eps_start
        self.full_agent_communication = args.full_agent_communication
        self.full_receptive_field = args.full_receptive_field
        self.game_name = args.game_name
        self.gamma = args.gamma
        self.gat_average_last = args.gat_average_last
        self.gat_n_heads = args.gat_n_heads
        self.graph_layer_type = args.graph_layer_type.upper()
        self.lr = args.lr
        self.max_num_episodes = args.max_num_episodes
        self.max_num_steps = args.max_num_steps
        self.max_steps_episode = args.max_steps_episode
        self.policy = args.policy.upper()
        self.replay_buffer_alpha = args.replay_buffer_alpha
        self.replay_buffer_beta = args.replay_buffer_beta
        self.replay_buffer_size = args.replay_buffer_size
        self.rgcn_n2_relations = args.rgcn_n2_relations
        self.rgcn_num_bases = args.rgcn_num_bases
        self.sparse_rewards = args.sparse_rewards
        self.target_update = args.target_update
        self.trr_coef = args.trr_coef
        self.weight_decay = args.weight_decay

        userhome = expanduser("~")
        self.log_dir = path_join(userhome, 'dodonet', 'smac')

        self.checkpoint_file = path_join(self.log_dir, 'checkpoints', self.checkpoint_name + '.pth')

        if isfile(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            self.training_datetime = checkpoint['training_datetime']
            self.step_num = checkpoint['total_steps']
            self.episode_num = checkpoint['n_episodes']
        else:
            self.step_num = 0
            self.episode_num = 0
            self.training_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        training_steps_to_go = args.max_num_steps - self.step_num
        if training_steps_to_go <= 0:
            print("Number of training steps achieved or surpassed. EXITING.")
            exit(0)

        tensorboard_dir = path_join(self.log_dir, 'tensorboard', self.training_datetime)
        self.summary_writer = SummaryWriter(tensorboard_dir)
        self.device = torch.device('cuda') if args.cuda_enable else torch.device('cpu')

        self.min_memory_size = min(self.replay_buffer_size, args.batch_size * 50)
        self.replay_buffer_save_interval = min(self.replay_buffer_size, self.min_memory_size * 3)

        self.replay_buffer_file = path_join(
            self.log_dir, 'replay_buffers', args.checkpoint_name + '.obj'
        )
        if isfile(self.replay_buffer_file):
            print('Replay buffer found on {}. LOADING...'.format(self.replay_buffer_file))
            self.memory = torch.load(self.replay_buffer_file)
            print('Replay buffer of size {} loaded from memory'.format(len(self.memory)))
        else:
            self.memory = PrioritizedReplayBuffer(
                self.replay_buffer_size, alpha=self.replay_buffer_alpha
            )

        memory_size_to_fill = self.min_memory_size - len(self.memory)
        total_steps = training_steps_to_go + memory_size_to_fill
        self.pbar = tqdm(range(total_steps))

        print(
            'Replay buffer has {} transitions stored, {} to fill.'.format(
                len(self.memory), max(0, memory_size_to_fill)
            )
        )
        print(
            '{} steps of training performed, {} to go.'.format(
                self.step_num, max(0, training_steps_to_go)
            )
        )

    def maybe_backup_buffer(self):
        if self.step_num % self.replay_buffer_save_interval == 0 and len(
            self.memory
        ) >= self.replay_buffer_save_interval:
            print('Saving a sample of the replay buffer to file...')
            # save a sample of the replay buffer to a file
            torch.save(self.memory.copy(self.replay_buffer_save_interval), self.replay_buffer_file)

    def log_episode(
        self,
        step_start,
        time_start,
        episode_reward: float = None,
        total_reward: float = None,
        battles_won: int = None,
        win_rate: float = None
    ):
        if episode_reward is not None:
            self.summary_writer.add_scalar('episode/episode_reward', episode_reward, self.step_num)
        if total_reward is not None:
            self.summary_writer.add_scalar('episode/total_reward', total_reward, self.step_num)
        if win_rate is not None:
            self.summary_writer.add_scalar('episode/win_rate', win_rate, self.step_num)
        if battles_won is not None:
            self.summary_writer.add_scalar('episode/battles_won', battles_won, self.step_num)
        self.summary_writer.add_scalar(
            'episode/num_steps', self.step_num - step_start, self.step_num
        )
        self.summary_writer.add_scalar(
            'episode/time_secs', (datetime.now() - time_start).total_seconds(), self.step_num
        )
