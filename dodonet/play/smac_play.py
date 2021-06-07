from datetime import datetime
from os.path import join as path_join

import torch

from dodonet.algorithms import MultiAgentDQN
from dodonet.encoding import LabelEncoder
from dodonet.envs import SMACEnv
from dodonet.training.trainer import Trainer


def run():
    trainer = Trainer(game_type=Trainer.GameType.SMAC)
    trainer.initialize()

    replay_dir = path_join(trainer.log_dir, 'game_replays', trainer.game_name)
    env = SMACEnv(
        map_name=trainer.game_name,
        replay_dir=replay_dir,
        reward_sparse=trainer.sparse_rewards,
        device=trainer.device
    )
    env_info = env.get_env_info()

    initialized = False
    while trainer.episode_num < trainer.max_num_episodes:
        env.reset()

        if not initialized:
            unit_types = env.get_unit_types()
            n_agents = env_info["n_agents"]
            n_actions = env_info["n_actions"]
            n_agent_features = env_info["n_agent_features"]
            n_enemy_features = env_info["n_enemy_features"]

            node_types = list(LabelEncoder(unit_types).transform(unit_types))
            unique_node_types = sorted(list(set(node_types)))

            agent_types = sorted(list(set(node_types[:n_agents])))

            features_by_node_type = [
                n_agent_features - env.unit_type_bits if nt in agent_types else n_enemy_features -
                env.unit_type_bits for nt in unique_node_types
            ]
            actions_by_node_type = [
                n_actions if nt in agent_types else 0 for nt in unique_node_types
            ]

            # notice the 'False' in the arguments, which will
            # configure the agent for play mode, not train mode
            model = MultiAgentDQN(
                trainer.checkpoint_file, trainer.summary_writer, trainer.policy,
                trainer.max_num_steps, trainer.batch_size, trainer.lr, trainer.weight_decay,
                trainer.trr_coef, trainer.checkpoint_save_secs, trainer.graph_layer_type,
                trainer.full_agent_communication, trainer.full_receptive_field, trainer.gat_n_heads,
                trainer.gat_average_last, trainer.rgcn_n2_relations, trainer.rgcn_num_bases,
                trainer.device, node_types, agent_types, features_by_node_type,
                actions_by_node_type, False, trainer.gamma, trainer.eps_start, trainer.eps_end,
                trainer.target_update
            )
            node_types = torch.tensor(node_types, device=trainer.device)

            initialized = True

        episode_reward = torch.zeros(model.n_agents, requires_grad=False)
        trainer.episode_num += 1

        step_start = trainer.step_num
        time_start = datetime.now()

        done = False
        while not done:
            trainer.step_num += 1

            # observe new state
            state_dict = env.get_state_dict()
            adj_matrix = env.get_visibility_matrix()
            current_state = model.gen_state(
                state_dict, env.unit_type_bits, adj_matrix, node_types
            ) if not done else None

            # Select and perform an action
            actions = model.act(
                model.policy_net, current_state, env.get_avail_actions(), trainer.step_num
            )
            reward, done, _ = env.step(actions)
            reward = torch.tensor([reward] * model.n_agents, dtype=torch.float)

            trainer.pbar.update()

            episode_reward += reward

        trainer.log_episode(
            step_start, time_start, episode_reward.mean(), None,
            env.get_stats()['battles_won'],
            env.get_stats()['win_rate']
        )
    env.close()


if __name__ == "__main__":
    run()
