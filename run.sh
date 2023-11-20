#!/bin/bash

source entrypoint.sh
source scripts/gcloud.sh
source scripts/ssh.sh

#############################################
# Docker
#############################################

DOCKER_REPO="gcr.io/$GCP_PROJECT_NAME"
DOCKER_IMAGE_NAME="$DOCKER_REPO/${PROJECT_NAME}:${BRANCH_NAME}"
DOCKER_RUN_ARGS="--shm-size=16g --rm"
DOCKER_VOLUMES="-v /var/tmp/out:/app/out/ -v /var/tmp/data:/app/data/ -v /var/tmp/cache:/root/.cache"
DOCKER_ENV=""

set-gpu() {
  export GPUS="$@"
  echo "GPUS=$GPUS"
}

set-daemon() {
  export DOCKER_RUN_ARGS="-d"
}

docker-build() {
  platform=$(uname -m)
  if [ "$platform" = "x86_64" ]; then
    docker build -t $DOCKER_IMAGE_NAME .
  elif [ "$platform" = "arm64" ]; then
    docker buildx build --platform=linux/amd64,linux/arm64 -t $DOCKER_IMAGE_NAME .
  fi
}

docker-push() {
  docker push $DOCKER_IMAGE_NAME
}

docker-build-push() {
  docker-build && docker-push
}

ssh-docker-rm() {
  ssh-cmd "docker kill \$(docker ps -aq)"
}

docker-run() {
  docker run ${DOCKER_VOLUMES} ${DOCKER_ENV} -it ${DOCKER_RUN_ARGS} ${DOCKER_IMAGE_NAME} ${@:-$DOCKER_COMMAND}
}

docker-run-gpu(){
  docker run ${DOCKER_VOLUMES} ${DOCKER_ENV} -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPUS:-all} ${DOCKER_RUN_ARGS} ${DOCKER_IMAGE_NAME} ${@:-$DOCKER_COMMAND}
}

ssh-docker-run() {
  ssh-cmd "docker run ${DOCKER_VOLUMES} ${DOCKER_ENV} -it --pull=always --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPUS:-all} ${DOCKER_RUN_ARGS} ${DOCKER_IMAGE_NAME} ${@:-$DOCKER_COMMAND}"
}

git-commit-hash() {
  git rev-parse HEAD
}


#############################################
# Commands
#############################################

set-cmd() {
  export DOCKER_COMMAND="$@"
  echo "DOCKER_COMMAND=$DOCKER_COMMAND"
}

set-docker-env() {
  export DOCKER_ENV="$@"
  echo "DOCKER_ENV=$DOCKER_ENV"
}

set-docker-volumes() {
  export DOCKER_VOLUMES="$@"
  echo "DOCKER_VOLUMES=$DOCKER_VOLUMES"
}

run-test() {
  set-cmd-train --test_mode_enabled=1
  docker-run
}

#############################################
# Tasks
#############################################

run-cmd() {
  ${DOCKER_COMMAND}
}

set-task(){
  set-cmd pyrun $@ --version $(git-commit-hash)
}

agent(){
  set-task train.py agent --sweep_id=$(cat ${WANDB_SWEEP_ID_FILE}) $@
}

agent-test(){
  agent --test_mode=1
  db && dr
}


train(){
  set-task train.py train $@
}

train-test(){
  train --test_mode=1
  db && dr
}

sweep() {
  set-task train.py sweep
  run-cmd
}

benchmark(){
  set-task benchmarks.py benchmark $@
}

benchmark-test(){
  benchmark --wandb_enabled=0 --num_len=1 --qry_limit=1 --ref_limit=1 
  db && dr
}

run-local(){
  db && drg
}

run-ssh(){
  dbp && sdr
}

run-cloud(){
  dbp && gcloud-create-one-gpu-try-all-zones
}

run-cloud-mult(){
  dbp
  for i in $(seq 1 $1); do
    gcloud-create-one-gpu-try-all-zones
  done
}

figures(){
  pyrun figures.py $@
}

figures-all(){
  set-task figures.py all
}

data-create() {
  set-task data.py create
}

data-upload() {
  set-task data.py upload
}

timing(){
  set-task timing.py om2seq
}



#############################################
# Aliases
#############################################

alias rt="run-test"
alias sdr="ssh-docker-run"
alias s="ssh-cmd"
alias db="docker-build"
alias dbp="docker-build-push"
alias dps="docker-push"
alias dr="docker-run"
alias drg="docker-run-gpu"
alias gs="git status"
alias ga="git add ."
alias gc="git commit -am ."
alias gps="git push"
alias gpl="git pull"
alias gcp="gc && gps"
alias pt="pytest -n 4 -s -v tests"
alias gc-c2-60="set-c2-60 && gcloud-create"
alias reload="source ./run.sh"
alias gccs="gcloud cloud-shell ssh --authorize-session"
alias rs="run-ssh"

#############################################
# Setups
#############################################

set-a100
set-azserver

$@

