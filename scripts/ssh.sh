SSH_ARGS="-t"

ssh-cmd(){
  ssh $SSH_ARGS $SSH_HOST $@
}

ssh-nvtop() {
  ssh-cmd nvtop
}

scp-secrets(){
  set-labpc && scp -r ./secrets .env.local $SSH_HOST:~/$PROJECT_NAME
}

set-ssh-host(){
  export SSH_HOST=$1
  echo "SSH_HOST=$SSH_HOST"
}

set-labserver(){
  set-ssh-host $SSH_USER@$SSH_LABSERVER
}

set-labpc(){
  set-ssh-host $SSH_USER@$SSH_LABPC
}

set-azserver(){
  set-ssh-host $SSH_USER@$SSH_AZSERVER
}
