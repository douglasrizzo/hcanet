train_hmagnet() {
  if [[ $# < 3 || $# > 5 ]]; then
    echo "Invalid number of arguments (MAP COMMS MIXER [REPLAY] [STEPS]), where:"
    echo
    echo "      MAP: the name of a SMAC map"
    echo "    COMMS: {None, GAT, RGCN}"
    echo "    MIXER: {TRUE, FALSE}"
    echo "   REPLAY: {None, mean, max} (def. None)"
    echo "    STEPS: number of training steps (def. 1000000)"

    return 1
  fi

  for i in $(seq 5); do
    train_hmagnet_single_v2 $1 $2 $3 $i ${4:-None} ${5:-1000000}
  done
}

train_hmagnet_single() {
  if [[ $# < 3 || $# > 5 ]]; then
    echo "Invalid number of arguments (MAP COMMS MIXER NUMBER [STEPS]), where:"
    echo
    echo "       MAP: the name of a SMAC map"
    echo "     COMMS: {None, GAT, RGCN}"
    echo "     MIXER: {TRUE, FALSE}"
    echo "    NUMBER: integer to number this run"
    echo "     STEPS: number of training steps (def. 1000000)"

    return 1
  fi

  BATCH_SIZE=32
  N_BASES=2
  N_HEADS=3
  map=${1}
  comms=${2}
  mixer=${3}
  i=${4}

  GROUP=A_${map}_${comms}_${mixer}
  RUNNAME=${GROUP}_${i}

  STEPS=${5:-1000000}

  # --episode_priority {mean,max,median}

  com="tsp python -m hmagnet.training.train SMAC DDQN ${map} ${RUNNAME} --run_prefix ${GROUP} --batch_size ${BATCH_SIZE} --max_num_steps ${STEPS} --policy egreedy_decay --double_dqn --v2_state --use_rnn_action --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64"

  graph_pars=""
  mixer_pars=""

  if [[ $comms != "None" ]]; then
    graph_pars="--graph_layer_type ${comms} --rgcn_num_bases ${N_BASES} --gat_n_heads ${N_HEADS}"
  fi
  if [[ $mixer = "TRUE" ]]; then
    mixer_pars="--mixer vdn"
  fi
  eval "$com $graph_pars $mixer_pars"
}

train_hmagnet_single_v2() {
  if [[ $# < 3 || $# > 5 ]]; then
    echo "Invalid number of arguments (MAP COMMS MIXER NUMBER [STEPS]), where:"
    echo
    echo "       MAP: the name of a SMAC map"
    echo "     COMMS: {None, GAT, RGCN}"
    echo "     MIXER: {TRUE, FALSE}"
    echo "    NUMBER: integer to number this run"
    echo "     STEPS: number of training steps (def. 1000000)"

    return 1
  fi

  BATCH_SIZE=32
  N_BASES=2
  N_HEADS=3
  map=${1}
  comms=${2}
  mixer=${3}
  i=${4}

  GROUP=A_${map}_${comms}_${mixer}
  RUNNAME=${GROUP}_${i}

  STEPS=${5:-1000000}

  # --episode_priority {mean,max,median}

  com="tsp python -m hmagnet.training.train SMAC DDQN ${map} ${RUNNAME} --run_prefix ${GROUP} --batch_size ${BATCH_SIZE} --max_num_steps ${STEPS} --policy egreedy_decay --double_dqn --v2_state --use_rnn_action --share_encoding --share_action --act_encoding tanh --act_comms leakyrelu --act_action leakyrelu --device cuda --encoding_hidden 96 --comms_sizes 96,96 --action_hidden 64 --optimizer adam --save_replays --eval_episodes 10 --eval_interval 15000"

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
