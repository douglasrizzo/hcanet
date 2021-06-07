# # RGCN with different number of bases
# # GAT with different number of heads
for i in {1..10}; do
   for map in 3m 2s3z 3s5z 1c3s5z MMM MMM2; do
      tsp -L RGCN_${i}_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type RGCN --rgcn_num_bases ${i}
      tsp -L GAT_${i}_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type GAT --gat_n_heads ${i}
   done
done

# # Network with no communication
tsp -L None python -m hmagnet.training.train SMAC DDQN MMM dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run

for map in 3m 2s3z 3s5z 1c3s5z MMM MMM2; do
   # Network with parameter sharing, different maps
   tsp -L RGCN_Y_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type RGCN --rgcn_num_bases 2
   tsp -L GAT_Y_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type GAT --gat_n_heads 3
   tsp -L None_Y_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run

   # Share encoding module
   tsp -L RGCN_E_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type RGCN --rgcn_num_bases 2
   tsp -L GAT_E_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type GAT --gat_n_heads 3
   tsp -L None_E_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_encoding --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run

   # Share action module
   tsp -L RGCN_A_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type RGCN --rgcn_num_bases 2
   tsp -L GAT_A_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type GAT --gat_n_heads 3
   tsp -L None_A_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run

   # Network without parameter sharing, different maps
   tsp -L RGCN_N_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type RGCN --rgcn_num_bases 2
   tsp -L GAT_N_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run --graph_layer_type GAT --gat_n_heads 3
   tsp -L None_N_${map} python -m hmagnet.training.train SMAC DDQN ${map} dry_test_run --run_prefix dry_test --batch_size 32 --max_num_steps 10 --policy egreedy_decay --double_dqn --use_rnn_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --dry_run
done
