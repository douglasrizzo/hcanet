train_hcanet() {
  if [[ $# < 2 || $# > 4 ]]; then
    echo "Invalid number of arguments (MAP COMMS MIXER [STEPS]), where:"
    echo
    echo "      MAP: the name of a SMAC map"
    echo "    COMMS: {None, GAT, RGCN}"
    echo "    MIXER: {TRUE, FALSE}"
    echo "    STEPS: number of training steps (def. 5000000)"

    return 1
  fi

  train_hcanet_single_v2_small_eval_save_replay $1 $2 $3 1 ${4:-5000000}

  for i in $(seq 2 5); do
    train_hcanet_single_v2 $1 $2 $3 $i ${4:-5000000}
  done
}

train_hcanet_single_v2_small_eval_save_replay() {
  if [[ $# < 3 || $# > 5 ]]; then
    echo "Invalid number of arguments (MAP COMMS MIXER NUMBER [STEPS]), where:"
    echo
    echo "       MAP: the name of a SMAC map"
    echo "     COMMS: {None, GAT, RGCN}"
    echo "     MIXER: {TRUE, FALSE}"
    echo "    NUMBER: integer to number this run"
    echo "     STEPS: number of training steps (def. 5000000)"

    return 1
  fi

  BATCH_SIZE=32
  N_BASES=2
  N_HEADS=3
  map=${1}
  comms=${2}
  mixer=${3}
  i=${4}

  GROUP=C_${map}_${comms}_${mixer}
  RUNNAME=${GROUP}_${i}

  STEPS=${5:-5000000}

  com="tsp python -m hcanet.training.train SMAC DDQN ${map} ${RUNNAME} --run_prefix ${GROUP} --batch_size ${BATCH_SIZE} --max_num_steps ${STEPS} --policy egreedy_decay --double_dqn --v2_state --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms tanh --act_action tanh --device smart --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --optimizer adam --eval_interval 10000 --eps_start 1 --eps_end 0.1 --eps_anneal_time 600000 --save_replays --eval_episodes 10"

  graph_pars=""
  mixer_pars=""

  if [[ $comms != "None" ]]; then
    graph_pars="--graph_layer_type ${comms} --rgcn_num_bases ${N_BASES} --gat_n_heads ${N_HEADS}"
  fi
  if [[ $mixer = "TRUE" ]]; then
    mixer_pars="--mixer vdn"
  fi
  eval "$com $graph_pars $mixer_pars $replay_pars"
}

train_hcanet_single_v2() {
  if [[ $# < 3 || $# > 5 ]]; then
    echo "Invalid number of arguments (MAP COMMS MIXER NUMBER [STEPS]), where:"
    echo
    echo "       MAP: the name of a SMAC map"
    echo "     COMMS: {None, GAT, RGCN}"
    echo "     MIXER: {TRUE, FALSE}"
    echo "    NUMBER: integer to number this run"
    echo "     STEPS: number of training steps (def. 5000000)"

    return 1
  fi

  BATCH_SIZE=32
  N_BASES=2
  N_HEADS=3
  map=${1}
  comms=${2}
  mixer=${3}
  i=${4}

  GROUP=C_${map}_${comms}_${mixer}
  RUNNAME=${GROUP}_${i}

  STEPS=${5:-5000000}

  com="tsp python -m hcanet.training.train SMAC DDQN ${map} ${RUNNAME} --run_prefix ${GROUP} --batch_size ${BATCH_SIZE} --max_num_steps ${STEPS} --policy egreedy_decay --double_dqn --v2_state --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device smart --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --optimizer adam --eval_interval 10000 --eps_start 1 --eps_end 0.1 --eps_anneal_time 600000 --eval_episodes 32"

  graph_pars=""
  mixer_pars=""

  if [[ $comms != "None" ]]; then
    graph_pars="--graph_layer_type ${comms} --rgcn_num_bases ${N_BASES} --gat_n_heads ${N_HEADS}"
  fi
  if [[ $mixer = "TRUE" ]]; then
    mixer_pars="--mixer vdn"
  fi
  eval "$com $graph_pars $mixer_pars $replay_pars"
}
