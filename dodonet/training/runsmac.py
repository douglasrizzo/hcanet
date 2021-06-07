from datetime import datetime
from os.path import join as path_join

import torch

from dodonet.algorithms import MultiAgentActorCritic, MultiAgentDQN
from dodonet.encoding import LabelEncoder
from dodonet.envs import SMACEnv
from dodonet.training.trainer import Trainer


class SMACRunner:

    def __init__(self):
        self.trainer = Trainer(game_type=Trainer.GameType.SMAC)
        self.trainer.initialize()

        replay_dir = path_join(self.trainer.log_dir, 'game_replays', self.trainer.game_name)
        self.env = SMACEnv(
            map_name=self.trainer.game_name,
            replay_dir=replay_dir,
            reward_sparse=self.trainer.sparse_rewards,
            device=self.trainer.device
        )
        env_info = self.env.get_env_info()

        self.env.reset()
        # this information can only be acquired after the environment is initialized
        unit_types = self.env.get_unit_types()
        n_agents = env_info["n_agents"]
        n_actions = env_info["n_actions"]
        n_agent_features = env_info["n_agent_features"]
        n_enemy_features = env_info["n_enemy_features"]

        node_types = list(LabelEncoder(unit_types).transform(unit_types))
        unique_node_types = sorted(list(set(node_types)))
        agent_types = sorted(list(set(node_types[:n_agents])))

        self.trainer.entropies = torch.zeros(n_agents)
        self.node_types = torch.tensor(node_types, device=self.trainer.device)

        features_by_node_type = [
            n_agent_features - self.env.unit_type_bits if nt in agent_types else n_enemy_features -
            self.env.unit_type_bits for nt in unique_node_types
        ]
        actions_by_node_type = [n_actions if nt in agent_types else 0 for nt in unique_node_types]

        self.make_agent(node_types, agent_types, features_by_node_type, actions_by_node_type)

    def run(self):
        raise NotImplementedError

    def make_agent(self, node_types, agent_types, features_by_node_type, actions_by_node_type):
        raise NotImplementedError


class OffPolicySMACRunner(SMACRunner):

    def make_agent(self, node_types, agent_types, features_by_node_type, actions_by_node_type):
        self.model = MultiAgentDQN(
            self.trainer.checkpoint_file, self.trainer.summary_writer, self.trainer.policy,
            self.trainer.max_num_steps, self.trainer.batch_size, self.trainer.lr,
            self.trainer.weight_decay, self.trainer.trr_coef, self.trainer.checkpoint_save_secs,
            self.trainer.graph_layer_type, self.trainer.full_agent_communication,
            self.trainer.full_receptive_field, self.trainer.coalesce_state_obs,
            self.trainer.gat_n_heads, self.trainer.gat_average_last, self.trainer.rgcn_n2_relations,
            self.trainer.rgcn_num_bases, self.trainer.device, node_types, agent_types,
            features_by_node_type, actions_by_node_type, True, self.trainer.gamma,
            self.trainer.eps_start, self.trainer.eps_end, self.trainer.target_update
        )

    def run(self):
        while self.trainer.step_num < self.trainer.max_num_steps:
            self.env.reset()

            current_state = self.env.get_graph_state(
                self.node_types,
                self.model.agent_types if self.model.full_agent_communication else None
            )

            episode_reward = torch.zeros(self.model.n_agents, requires_grad=False)
            step_start = self.trainer.step_num
            episode_steps = 0
            time_start = datetime.now()

            done = False
            while not done:
                if self.trainer.max_steps_episode is not None and episode_steps >= self.trainer.max_steps_episode:
                    break

                episode_steps += 1
                self.trainer.step_num += 1
                # Select and perform an action
                actions = self.model.act(
                    current_state, self.env.get_avail_actions(), self.trainer.step_num
                )
                reward, done, _ = self.env.step(actions)
                reward = torch.tensor([reward] * self.model.n_agents, dtype=torch.float)

                self.trainer.pbar.update()

                # observe new state
                next_state = self.env.get_graph_state(
                    self.node_types,
                    self.model.agent_types if self.model.full_agent_communication else None
                ) if not done else None

                # Store the transition in memory
                self.trainer.memory.add(current_state, actions, next_state, reward, float(done))
                self.trainer.maybe_backup_buffer()

                # Perform one step of the optimization (on the target network)
                if len(self.trainer.memory) >= self.trainer.min_memory_size:
                    self.model.optimize(self.trainer)

                    net_weights = {
                        'policy_net_state_dict': self.model.policy_net.state_dict(),
                        'target_net_state_dict': self.model.target_net.state_dict()
                    }
                    self.model.maybe_save_checkpoint(
                        net_weights, self.trainer.step_num, self.trainer.training_datetime
                    )

                # Update the target network, copying all
                # weights and biases from the policy network
                if self.trainer.step_num % self.trainer.target_update == 0:
                    self.model.update_target_net()

                # Move to the next state
                current_state = next_state
                episode_reward += reward

            self.trainer.log_episode(
                step_start, time_start, episode_reward.mean(), None,
                self.env.get_stats()['battles_won'],
                self.env.get_stats()['win_rate']
            )
        self.env.close()


class OnPolicySMACRunner(SMACRunner):

    def make_agent(self, node_types, agent_types, features_by_node_type, actions_by_node_type):
        self.model = MultiAgentActorCritic(
            self.trainer.checkpoint_file, self.trainer.summary_writer, self.trainer.policy,
            self.trainer.max_num_steps, self.trainer.batch_size, self.trainer.lr,
            self.trainer.weight_decay, self.trainer.trr_coef, self.trainer.checkpoint_save_secs,
            self.trainer.graph_layer_type, self.trainer.full_agent_communication,
            self.trainer.full_receptive_field, self.trainer.coalesce_state_obs,
            self.trainer.gat_n_heads, self.trainer.gat_average_last, self.trainer.rgcn_n2_relations,
            self.trainer.rgcn_num_bases, self.trainer.device, node_types, agent_types,
            features_by_node_type, actions_by_node_type, True, self.trainer.gamma,
            self.trainer.eps_start, self.trainer.eps_end, self.trainer.target_update
        )

    def run(self):
        while self.trainer.step_num < self.trainer.max_num_steps:
            self.env.reset()

            current_state = self.env.get_graph_state(
                self.node_types,
                self.model.agent_types if self.model.full_agent_communication else None
            )

            episode_reward = torch.zeros(self.model.n_agents, requires_grad=False)
            step_start = self.trainer.step_num
            episode_steps = 0
            time_start = datetime.now()

            done = False
            while not done:
                if self.trainer.max_steps_episode is not None and episode_steps >= self.trainer.max_steps_episode:
                    break

                episode_steps += 1
                self.trainer.step_num += 1
                # Select and perform an action
                agent_indices, probs, values = self.model.policy_net(current_state)

                actions, dists = self.model.act(
                    agent_indices, probs, self.env.get_avail_actions(), self.model.device
                )

                reward, done, _ = self.env.step(actions)
                reward = torch.tensor([reward] * self.model.n_agents, dtype=torch.float)

                log_probs = torch.tensor([dists[i].log_prob(actions[i]) for i in range(dists)])

                self.trainer.entropy = self.trainer.entropy + torch.tensor(
                    [dist.entropy().mean() for dist in dists]
                )

                # we only append the state if TRR is to be calculated, otherwise try to save some memory, lol
                self.trainer.on_policy_trajectories.append(
                    (
                        log_probs, values, reward, done,
                        current_state if self.trainer.trr_coef != 0 else None
                    )
                )

                self.trainer.pbar.update()

                # observe new state
                next_state = self.env.get_graph_state(
                    self.node_types,
                    self.model.agent_types if self.model.full_agent_communication else None
                ) if not done else None

                # Perform one step of the optimization
                if done or len(
                    self.trainer.max_trajectory_length is not None and
                    self.trainer.on_policy_trajectories
                ) >= self.trainer.max_trajectory_length:
                    _ = self.model.optimize(self.trainer.on_policy_trajectories)
                    self.trainer.on_policy_trajectories = []

                    net_weights = {
                        'actor_critic_net_state_dict': self.model.policy_net.state_dict()
                    }
                    self.model.maybe_save_checkpoint(
                        net_weights, self.trainer.step_num, self.trainer.training_datetime
                    )

                # Move to the next state
                current_state = next_state
                episode_reward += reward

            self.trainer.log_episode(
                step_start, time_start, episode_reward.mean(), None,
                self.env.get_stats()['battles_won'],
                self.env.get_stats()['win_rate']
            )
        self.env.close()


if __name__ == "__main__":
    OffPolicySMACRunner().run()
