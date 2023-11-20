#!/bin/bash

install_docker(){
  sudo apt-get update
  sudo apt-get install ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  sudo apt-get update
  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
  sudo usermod -aG docker $USER
  newgrp docker
  docker run hello-world
}

install_buildx(){
  # https://vikaspogu.dev/posts/docker-buildx-setup/
  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
}

install_nvidia_docker(){
  # sudo ubuntu-drivers autoinstall
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

  sudo apt-get update && sudo apt-get install -y nvidia-docker2
  sudo systemctl restart docker
}

install_gcloud(){
  sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
  sudo apt-get update && sudo apt-get install google-cloud-cli

  gcloud auth activate-service-account --key-file=-
  gcloud auth configure-docker gcr.io
}

setup_gpu_machine(){
  install_docker
  install_nvidia_docker
  install_gcloud
}

setup_macos(){
  install_docker
  install_buildx
  install_gcloud
}

$@