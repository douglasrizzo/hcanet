from datetime import datetime

import torch

import pommerman
from dodonet.algorithms import MultiAgentDQN
from dodonet.envs import PommermanEnv
from dodonet.training.trainer import Trainer
from pommerman import agents


class PommermanRunner:

    def __init__(self):
        """Simple function to bootstrap a game"""
        # Print all possible environments in the Pommerman registry
        print(pommerman.REGISTRY)

        self.my_team = [PommermanEnv.BrainlessAgent(), PommermanEnv.BrainlessAgent()]

        # Create a set of agents (exactly four)
        agent_list = [
            self.my_team[0],
            agents.SimpleAgent(),
            self.my_team[1],
            agents.SimpleAgent(),
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]
        self.trainer = Trainer(game_type=Trainer.GameType.POMMERMAN)
        self.trainer.initialize()

        self.env = pommerman.make(self.trainer.game_name, agent_list)
        self.players_t1 = [0, 2]
        node_types = [0, 1, 0, 1]
        agent_types = [0]
        features_by_node_type = [PommermanEnv.get_feacture_vector_len()] * 2
        actions_by_node_type = [self.env.action_space.n, 0]

        self.make_agent(actions_by_node_type, agent_types, features_by_node_type, node_types)

    def make_agent(self, actions_by_node_type, agent_types, features_by_node_type, node_types):
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
        total_reward = 0
        battles_won = 0

        while self.trainer.step_num < self.trainer.max_num_steps:
            self.trainer.episode_num += 1
            raw_state = self.env.reset()
            current_state = self.model.gen_state(raw_state, self.players_t1)

            step_start = self.trainer.step_num
            time_start = datetime.now()
            episode_steps = 0

            done = False
            while not done:
                if self.trainer.max_steps_episode is not None and episode_steps >= self.trainer.max_steps_episode:
                    break

                # env.render()
                episode_steps += 1
                self.trainer.step_num += 1

                _, acts = self.model.act(current_state, self.trainer.step_num)
                for i in range(len(self.my_team)):
                    self.my_team[i].action = int(acts[i])

                actions = self.env.act(raw_state)

                # observe new state
                next_raw_state, reward, done, info = self.env.step(actions)

                # get only the reward for the players in my team
                reward = torch.tensor([reward[p] for p in self.players_t1], dtype=torch.float)
                next_state = self.model.gen_state(
                    next_raw_state, self.players_t1
                ) if not done else None
                self.trainer.pbar.update()

                # Store the transition in memory
                self.trainer.memory.add(current_state, actions, next_state, reward, float(done))
                self.trainer.maybe_backup_buffer()

                # Move to the next state
                current_state = next_state

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

                total_reward += reward

            if info['result'].value == 0 and info['winners'] == self.my_team:
                battles_won += 1

            # win_rate = battles_won / self.trainer.episode_num

            self.trainer.log_episode(step_start, time_start, None, total_reward, battles_won, None)


if __name__ == '__main__':
    PommermanRunner().run()
