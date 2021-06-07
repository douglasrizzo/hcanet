import os
import wandb
from abc import ABC, abstractmethod
from datetime import datetime
from os import path

import torch as th
from tqdm import tqdm

from ..controllers import (BaseController, DQNController, DRLController, RandomController)
from ..encoding import LabelEncoder
from ..envs import SMACEnv
from ..rl.replay import EpisodeReplayBuffer, PrioritizedEpisodeReplayBuffer, Transition
from ..training import TrainingConfig

# th.autograd.set_detect_anomaly(True)


class Runner(ABC):

   def __init__(self, trainer: TrainingConfig):
      self.training_config = trainer
      self.controller: BaseController

   @abstractmethod
   def make_agent(self,
                  node_types,
                  agent_types,
                  features_by_node_type,
                  actions_by_node_type,
                  encoding_output_size,
                  graph_module_sizes,
                  action_hidden_size):
      raise NotImplementedError

   @abstractmethod
   def run(self):
      raise NotImplementedError


class SMACRunner(Runner, ABC):

   def __init__(self, trainer: TrainingConfig):
      super().__init__(trainer)

      self.checkpoint_file = path.join(trainer.log_dir, 'checkpoint.pth')
      resume_run = path.isfile(self.checkpoint_file)

      self.step_num = 0
      self.episode_num = 0
      previous_step_num = 0

      if resume_run:
         print("Checkpoint file found, loading initial data...")
         checkpoint = th.load(self.checkpoint_file)
         previous_step_num = checkpoint['total_steps']
         self.episode_num = checkpoint['n_episodes']

      self.step_num += previous_step_num

      if not trainer.resume_run:
         trainer.max_num_steps += previous_step_num
      if self.step_num >= trainer.max_num_steps:
         print("Number of training steps achieved or surpassed. EXITING.")
         exit(0)

      if not trainer.dry_run:
         if not path.exists(trainer.log_dir):
            os.makedirs(trainer.log_dir)

         args_to_log = trainer.get_loggable_args()
         wandb_config = {}
         for key in args_to_log:
            for subkey in args_to_log[key]:
               wandb_config[subkey[0]] = subkey[1]

         wandb.init(project='hcanet',
                    name=trainer.run_name,
                    id=trainer.run_name,
                    dir='/tmp/wandb',
                    resume=resume_run,
                    config=wandb_config,
                    group=trainer.run_prefix)

      if trainer.episode_priority is None:
         self.memory = EpisodeReplayBuffer(trainer.replay_buffer_size)
      else:
         self.memory = PrioritizedEpisodeReplayBuffer(
             trainer.replay_buffer_size,
             trainer.replay_buffer_alpha,
         )

      self.pbar = tqdm(initial=self.step_num - previous_step_num,
                       total=trainer.max_num_steps - previous_step_num,
                       smoothing=0)

      replay_dir = path.join(trainer.log_dir, 'game_replays', trainer.game_name)
      self.env = SMACEnv(map_name=trainer.game_name,
                         replay_dir=replay_dir,
                         reward_sparse=trainer.sparse_rewards)
      env_info = self.env.get_env_info()

      self.env.reset()
      # this information can only be acquired after the environment is initialized
      unit_types = self.env.get_unit_types()
      n_agents = env_info["n_agents"]
      n_actions = env_info["n_actions"]
      # n_agent_features = len(env_info["agent_features"])
      # n_enemy_features = len(env_info["enemy_features"])
      v2_obs_shape = env_info["obs_shape"]

      # get unit types from the environment
      # normalize using label encoder
      # ignore non-agent unit types
      self.node_types = list(LabelEncoder(unit_types).transform(unit_types))
      self.node_types = th.tensor(self.node_types[:n_agents], device=trainer.device).int()

      agent_types = self.node_types.unique().tolist()
      features_by_node_type = [v2_obs_shape] * len(agent_types)
      actions_by_node_type = [n_actions] * len(agent_types)

      self.controller = self.make_agent(self.node_types.tolist(),
                                        agent_types,
                                        features_by_node_type,
                                        actions_by_node_type,
                                        trainer.encoding_hidden_size,
                                        trainer.comms_sizes,
                                        trainer.action_hidden_size)

      if trainer.dry_run:
         exit(0)

   def sample_from_memory(self):
      return self.memory.sample(
          self.batch_size) if not self.memory.is_prioritized else self.memory.sample(
              self.batch_size, self.replay_buffer_beta)

   def maybe_backup_buffer(self):
      if self.step_num % self.replay_buffer_save_interval == 0 and len(
          self.memory) >= self.replay_buffer_save_interval:
         print('Saving a sample of the replay buffer to file...')
         th.save(self.memory.copy(self.replay_buffer_save_interval), self.replay_buffer_file)

   def log_episode(self, things_to_log, prefix='episode'):
      # add the prefix to arg names
      loggers_poggers = {}
      for key in things_to_log:
         loggers_poggers[prefix + '/' + key] = things_to_log[key]

      wandb.log(loggers_poggers, step=self.step_num)


class OffPolicySMACRunner(SMACRunner):

   def make_agent(self,
                  node_types,
                  agent_types,
                  features_by_node_type,
                  actions_by_node_type,
                  encoding_output_size,
                  graph_module_sizes,
                  action_hidden_size):
      return DQNController(self.checkpoint_file,
                           self.training_config.action_module,
                           self.training_config.policy,
                           self.training_config.max_num_steps,
                           self.training_config.batch_size,
                           self.training_config.optimizer,
                           self.training_config.lr,
                           self.training_config.weight_decay,
                           self.training_config.rmsprop_alpha,
                           self.training_config.rmsprop_eps,
                           self.training_config.trr_coef,
                           self.training_config.checkpoint_save_secs,
                           self.training_config.graph_layer_type,
                           self.training_config.share_encoding,
                           self.training_config.share_comms,
                           self.training_config.share_action,
                           self.training_config.full_agent_communication,
                           self.training_config.full_receptive_field,
                           self.training_config.gat_n_heads,
                           self.training_config.gat_average_last,
                           self.training_config.rgcn_n2_relations,
                           self.training_config.rgcn_num_bases,
                           self.training_config.rgcn_fast,
                           self.training_config.device,
                           node_types,
                           agent_types,
                           features_by_node_type,
                           actions_by_node_type,
                           self.training_config.training_mode,
                           self.training_config.data_parallel,
                           self.training_config.act_encoding,
                           self.training_config.act_comms,
                           self.training_config.act_action,
                           self.training_config.use_rnn_encoding,
                           self.training_config.use_rnn_action,
                           self.training_config.gamma,
                           self.training_config.eps_start,
                           self.training_config.eps_end,
                           self.training_config.eps_anneal_time,
                           self.training_config.target_update,
                           self.training_config.double_dqn,
                           self.training_config.mixer,
                           encoding_output_size=encoding_output_size,
                           graph_module_sizes=graph_module_sizes,
                           action_hidden_size=action_hidden_size)

   def run(self):
      last_eval = 0
      training_start = datetime.now()
      while self.step_num < self.training_config.max_num_steps:
         step_start = self.step_num
         time_start = datetime.now()

         episode, episode_reward, info = self.play_episode()

         # training_mode is true when not in eval mode
         # TODO this variable seems useless
         if self.training_config.training_mode:
            # Store the transition in memory
            self.memory.add(episode)
            # self.trainer.maybe_backup_buffer()

            # Perform one step of the optimization (on the target network)
            if self.memory.can_sample(self.training_config.batch_size):
               self.controller.policy_net.train()
               self.controller.optimize(self.step_num, self.training_config, self.memory)
               self.controller.maybe_save_checkpoint(self.step_num)

            # Update the target network, copying all
            # weights and biases from the policy network
            if self.episode_num % self.training_config.target_update == 0:
               self.controller.update_target_net()

         things_to_log = {
             'episode_reward': episode_reward,
             'battles_won': self.env.get_stats()['battles_won'],
             'time_secs': (datetime.now() - time_start).total_seconds(),
             'num_steps': self.step_num - step_start}
         if 'dead_allies' in info:
            things_to_log['dead_allies'] = info['dead_allies']
            things_to_log['dead_enemies'] = info['dead_enemies']

         self.log_episode(things_to_log)

         # evaluation
         # only evaluate if has already been trained
         if self.memory.can_sample(
             self.training_config.batch_size
         ) and self.step_num - last_eval >= self.training_config.eval_interval:
            last_eval = self.step_num - (self.step_num % self.training_config.eval_interval)
            self.evaluate(n_episodes=self.training_config.eval_episodes)
            # release GPU cache alongside evaluation
            th.cuda.empty_cache()

      with open(path.join(self.training_config.log_dir, 'run_time.txt'), 'a') as f:
         f.write(str(datetime.now() - training_start))

      self.env.close()

   def play_episode(self):
      self.env.reset()
      current_state = self.env.get_graph_state(
          self.node_types,
          self.controller.agent_types if self.controller.full_agent_communication else None,
          v2=self.training_config.v2_state)

      self.episode_num += 1

      episode_reward = 0
      episode_steps = 0
      episode = []

      done = False
      with th.no_grad():
         self.controller.policy_net.eval()
         self.controller.policy_net.action_layer.init_hidden(1)
         while not done:
            episode_steps += 1
            self.step_num += 1

            # I did this and the network learned something
            # batch = [t.state.to(self.controller.device) for t in episode] + [current_state]
            # batch = Batch.from_data_list(batch)
            # q_vals = self.controller.policy_net(batch)
            q_vals = self.controller.policy_net(current_state.to(self.training_config.device))

            # Select and perform an action
            av_actions = self.env.get_avail_actions()
            actions = self.controller.act(q_vals[0], av_actions, self.step_num)

            # if isinstance(self.controller, MultiAgentActorCritic):
            #    actions = actions[0]

            reward, done, info = self.env.step(actions)

            self.pbar.update()

            # observe new state
            next_state = None if done else self.env.get_graph_state(
                self.node_types,
                self.controller.agent_types if self.controller.full_agent_communication else None,
                v2=self.training_config.v2_state)

            # pass everything to CPU for storage
            # NOTE I don't know if this actually saves GPU memory
            for i, _ in enumerate(av_actions):
               av_actions[i] = av_actions[i].cpu()
            episode.append(
                Transition(current_state.to(th.device('cpu')),
                           actions.cpu(),
                           reward,
                           float(done),
                           av_actions))

            # Move to the next state
            current_state = next_state
            episode_reward += reward
      return episode, episode_reward, info

   def evaluate(self, n_episodes=32, close_env=False):
      time_start = datetime.now()
      battles_won = dead_allies = dead_enemies = eval_reward = 0

      for _ in tqdm(range(n_episodes), desc='Ep.'):
         episode, episode_reward, info = self.play_episode()

         eval_reward += episode_reward
         if 'dead_allies' in info:
            dead_allies += info['dead_allies']
            dead_enemies += info['dead_enemies']
         if 'battle_won' in info:
            battles_won += info['battle_won']

         if self.training_config.save_replays:
            self.env.save_replay()

      things_to_log = {
          'episode_reward': (eval_reward / n_episodes),
          'battles_won': battles_won / n_episodes,
          'time_secs': (datetime.now() - time_start).total_seconds() / n_episodes}
      if 'dead_allies' in info:
         things_to_log['dead_allies'] = dead_allies / n_episodes
         things_to_log['dead_enemies'] = dead_enemies / n_episodes

      self.log_episode(things_to_log, prefix='eval')

      if close_env:
         self.env.close()


class RandomSMACRunner(SMACRunner):

   def make_agent(self,
                  node_types,
                  agent_types,
                  features_by_node_type,
                  actions_by_node_type,
                  encoding_hidden_sizes=None,
                  encoding_output_size=None,
                  graph_hidden_sizes=None,
                  graph_output_size=None,
                  action_hidden_size=None):
      return RandomController(node_types,
                              agent_types,
                              features_by_node_type,
                              actions_by_node_type,
                              self.training_config.device)

   def run(self):
      self.controller.initialize()

      while self.step_num < self.training_config.max_num_steps:
         step_start = self.step_num
         time_start = datetime.now()
         self.episode_num += 1

         episode_reward = th.zeros(self.controller.n_agents, requires_grad=False)
         episode_steps = 0

         done = False
         while not done:
            if self.training_config.max_steps_episode is not None and episode_steps >= self.training_config.max_steps_episode:
               break

            episode_steps += 1
            self.step_num += 1
            # Select and perform an action

            actions = self.controller.act(self.env.get_avail_actions())

            reward, done, info = self.env.step(actions)
            reward = th.tensor([reward] * self.controller.n_agents, dtype=th.float)

            self.pbar.update()

            episode_reward += reward

         self.log_episode(step_start, time_start, episode_reward.mean(), info)

         if self.step_num < self.training_config.max_num_steps:
            self.env.reset()

      self.env.close()


if __name__ == "__main__":
   trainer = TrainingConfig()
   trainer.initialize()
   runner: Runner

   if trainer.game == TrainingConfig.GameType.SMAC:
      # if trainer.action_module in TrainingConfig.OFF_POLICY_METHODS:
      runner = OffPolicySMACRunner(trainer)
   # elif trainer.action_module == TrainingConfig.ActionModuleType.RANDOM:
   #    runner = RandomSMACRunner(trainer)
   else:
      raise ValueError("Game or action module type does not exist")

   try:
      runner.run()
   except (Exception, KeyboardInterrupt) as e:
      if isinstance(runner.controller, DRLController):
         print('Something happened, saving checkpoint...')
         runner.controller.save_checkpoint(runner.training_config.step_num)

      if not isinstance(e, KeyboardInterrupt):
         with open(path.join(trainer.log_dir, 'log.txt'), 'a') as f:
            import traceback
            f.write(str(e))
            f.write(traceback.format_exc())
         raise e
