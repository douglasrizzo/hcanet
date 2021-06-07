#!/bin/sh

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

SC2VERSION=4.10
SC2DIR=/home/dodo

pacman -S unzip

download_sc2() {
  echo 'Downloading StarCraft 2...'
  wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.${SC2VERSION}.zip && unzip -P iagreetotheeula SC2.${SC2VERSION}.zip -d ${SC2DIR}
  echo 'Finished downloading StarCraft 2!'
}

download_minigames() {
  echo 'Downloading minigames...'
  [ -f mini_games.zip ] || wget https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip && unzip -P iagreetotheeula -d ${SC2DIR}/StarCraftII/Maps mini_games.zip &
  echo 'Finished downloading minigames!'
}

download_s3maps() {
  echo 'Downloading S3 maps...'
  wget https://doc-10-a0-docs.googleusercontent.com/docs/securesc/29to5snd2i1d5rr56au13d5h1fjfukcm/f4c0auo5vpo1cahk54mirgilu85nq31v/1541116800000/08913212008526343923/11757355169451059804/1XahqdSkrPL6oV9IdXFNYhR8vQl4XMEGI?e=download -O S3Maps.zip && unzip -d ${SC2DIR}/StarCraftII/Maps S3Maps.zip
  echo 'Finished downloading S3 maps!'
}

download_ladder_maps() {
  echo 'Downloading ladder maps...'
  for fn in Ladder2017Season1.zip Ladder2017Season2.zip Ladder2017Season3_Updated.zip Ladder2017Season4.zip Ladder2018Season1.zip Ladder2018Season2_Updated.zip Ladder2018Season3.zip Ladder2018Season4.zip Ladder2019Season1.zip Ladder2019Season2.zip Ladder2019Season3.zip Melee.zip; do    
      [ -f $fn ] || wget http://blzdistsc2-a.akamaihd.net/MapPacks/$fn && unzip -P iagreetotheeula -d ${SC2DIR}/StarCraftII/Maps $fn &
  done

  wait
  echo 'Finished downloading ladder maps!'
}

download_smac_maps() {
  echo 'Downloading ladder maps...'
  [ -f SMAC_Maps.zip ] || wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip && unzip -d ${SC2DIR}/StarCraftII/Maps SMAC_Maps.zip &
  
  wait
  echo 'Finished downloading SMAC!'
}

download_replay_packs() {
  echo 'Downloading replay packs...'

  for fn in 3.16.1-Pack_1-fix.zip 3.16.1-Pack_2.zip; do    
      [ -f $fn ] || wget http://blzdistsc2-a.akamaihd.net/ReplayPacks/$fn && unzip -P iagreetotheeula -d ${SC2DIR}/StarCraftII/Replays $fn &
  done

  wait
  echo 'Finished downloading replay packs!'
}

download_sc2
download_minigames
download_s3maps
download_ladder_maps
download_smac_maps
# download_replay_packs

echo 'Giving ownership to the SC2 files to the user!'

[ -e ${SC2DIR}/StarCraftII ] && chown -R dodo:dodo ${SC2DIR}/StarCraftII

python -m pysc2.bin.map_list

echo 'Installation complete. You can test it by running the following commands:'
echo '    python -m pysc2.bin.agent --map Simple64'
echo '    python -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShards'
echo 'Also check if export SC2PATH="${SC2DIR}/StarCraftII" has been added to your .zshrc file'
