GCP_DISK_IMAGE=projects/computational-microscopy/global/images/nvidia-gpu-disk-image-1024gb
GCP_ZONE="us-central1-b"
GCP_DISK_SIZE=1024
GPU_COUNT=1

generate-filename() {
  echo $(date +"%Y-%m-%d-%H-%M-%S")-$(uuidgen | cut -c 1-8 | tr '[:upper:]' '[:lower:]')
}

set-c2-60(){
  export GCP_MACHINE_TYPE="c2-standard-60" GCP_ACCELERATOR_TYPE=""
}

set-c2-4(){
  export GCP_MACHINE_TYPE="c2-standard-4" GCP_ACCELERATOR_TYPE=""
}

set-n2d(){
  export GCP_MACHINE_TYPE="n2d-highcpu-224" GCP_ACCELERATOR_TYPE=""
}

set-l4(){
    export GCP_MACHINE_TYPE="g2-standard-4" GCP_ACCELERATOR_TYPE="nvidia-l4"
}

set-a100(){
    export GCP_MACHINE_TYPE="a2-highgpu-${GPU_COUNT}g" GCP_ACCELERATOR_TYPE="nvidia-tesla-a100" GCP_ZONE="us-central1-c"
}

set-a100-80gb(){
    export GCP_MACHINE_TYPE="a2-ultragpu-1g" GCP_ACCELERATOR_TYPE="nvidia-a100-80gb" GCP_ZONE="us-east4-c"
}

set-v100(){
    export GCP_MACHINE_TYPE="n1-standard-4" GCP_ACCELERATOR_TYPE="nvidia-tesla-v100" GCP_ZONE="us-central1-a"
}

startup-script(){
  cat <<EOF
#! /bin/bash
gcloud auth configure-docker --quiet
docker run ${DOCKER_VOLUMES} --pull=always ${DOCKER_RUN_ARGS} ${DOCKER_IMAGE_NAME} ${DOCKER_COMMAND}
gcloud compute instances delete $GCP_INSTANCE_NAME --zone=$GCP_ZONE --quiet
EOF
}

_gcloud-create(){
  gcloud compute instances create ${GCP_INSTANCE_NAME:?} --labels=user=${SSH_USER} --zone=${GCP_ZONE} --create-disk=auto-delete=yes,boot=yes,device-name=${GCP_INSTANCE_NAME},image=${GCP_DISK_IMAGE},mode=rw,size=${GCP_DISK_SIZE} --machine-type=${GCP_MACHINE_TYPE} --scopes=https://www.googleapis.com/auth/cloud-platform --verbosity=debug --metadata=startup-script="$(startup-script)" $@
}

gcloud-create() {
  GCP_INSTANCE_NAME="${GCP_MACHINE_TYPE}-$(generate-filename)"
  _gcloud-create  
}

gcloud-delete(){
  gcloud compute instances delete $@
}

gcloud-create-gpu() {
  GCP_INSTANCE_NAME="${GCP_MACHINE_TYPE}-${GCP_ACCELERATOR_TYPE}-$(generate-filename)"
  DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS} --gpus all"
  _gcloud-create --provisioning-model=SPOT --accelerator=count=${GPU_COUNT},type=${GCP_ACCELERATOR_TYPE}
}

gcloud-create-one-gpu-try-all-zones(){
  GCP_ZONES=$(gcloud compute zones list --format="value(name)" | grep "us-")
  echo "Available zones: $GCP_ZONES"

  # Convert the string into an array
  GCP_ZONES_ARRAY=($GCP_ZONES)

  # Shuffle the array
  SHUFFLED_ZONES=($(printf "%s
  " "${GCP_ZONES_ARRAY[@]}" | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>)'))

  # loop over SHUFFLED_ZONES now
  for GCP_ZONE in "${SHUFFLED_ZONES[@]}"; do
    echo trying: $GCP_ZONE
    if gcloud-create-gpu; then
      break
    fi
  done
}

gcloud-create-gpu-all-zones() {
  GCP_NUM_INSTANCES=2
  GCP_ZONES=$(gcloud compute zones list --format="value(name)")
  echo "Available zones: $GCP_ZONES"

  for GCP_ZONE in $GCP_ZONES; do
    for i in $(seq ${GCP_NUM_INSTANCES}); do
      gcloud-create &
      sleep .1
    done
  done
}

gcloud_auth_docker() {
  gcloud auth configure-docker --quiet gcr.io
}

gcloud-ssh() {
  gcloud compute ssh ${GCP_INSTANCE_NAME} -- $@
}

gcloud-attach(){
  # Get a list of all running instances
  echo "Fetching running instances..."
  INSTANCES=$(gcloud compute instances list --filter="status:RUNNING" --format="value(name,zone)")

  # Check if there are any running instances
  if [ -z "$INSTANCES" ]; then
      echo "No running instances found."
      exit 1
  fi

  # Convert the instances to an array
  readarray -t INSTANCE_ARRAY <<<"$INSTANCES"

  # Print the instances with their numbers
  echo "Running instances:"
  for i in "${!INSTANCE_ARRAY[@]}"; do
      echo "$((i+1)). ${INSTANCE_ARRAY[$i]}"
  done

  # Prompt the user to choose an instance
  read -p "Enter the number of the instance you want to attach to: " INSTANCE_NUMBER

  # Check if the input is a number and if the number is in the range of instances
  if [[ $INSTANCE_NUMBER =~ ^[0-9]+$ ]] && [ $INSTANCE_NUMBER -ge 1 ] && [ $INSTANCE_NUMBER -le ${#INSTANCE_ARRAY[@]} ]; then
      # If it is, calculate the array index and attach to the serial console
      INSTANCE_INDEX=$((INSTANCE_NUMBER-1))
      INSTANCE_DETAILS=(${INSTANCE_ARRAY[$INSTANCE_INDEX]})
      INSTANCE_NAME=${INSTANCE_DETAILS[0]}
      INSTANCE_ZONE=${INSTANCE_DETAILS[1]}
      echo "Checking serial port access for ${INSTANCE_NAME}..."

      # Check if serial-port-enable is set to 'yes'
      SERIAL_PORT_ACCESS=$(gcloud compute instances describe $INSTANCE_NAME --zone $INSTANCE_ZONE --format="value(metadata.items.serial-port-enable)")

      if [ "$SERIAL_PORT_ACCESS" != "yes" ]; then
          echo "Enabling serial port access for ${INSTANCE_NAME}..."
          gcloud compute instances add-metadata $INSTANCE_NAME --zone $INSTANCE_ZONE --metadata serial-port-enable=yes
      fi

      echo "Attaching to the serial console of ${INSTANCE_NAME}..."
      gcloud compute connect-to-serial-port $INSTANCE_NAME --zone=$INSTANCE_ZONE
  else
      # If not, print an error message and exit with a non-zero status code
      echo "Invalid number. Please enter a number between 1 and ${#INSTANCE_ARRAY[@]}."
      exit 1
  fi
}

gcloud-scripts(){
    # Fetch all running instances
  instances=$(gcloud compute instances list --filter="status=RUNNING" --format="value(name,zone)")

  # Loop over instances and fetch their startup script
  while IFS= read -r line; do
    instance_name=$(echo $line | awk '{print $1}')
    instance_zone=$(echo $line | awk '{print $2}')

    echo "Instance Name: $instance_name"
    echo "Instance Zone: $instance_zone"

    # Fetch startup script
    startup_script=$(gcloud compute instances describe $instance_name --zone $instance_zone --format='value(metadata.items.key.startup-script)')

    echo "Startup Script: "
    echo "$startup_script"
    echo "-------------"
  done <<< "$instances"
}

gcloud-tail(){
    #!/bin/bash

  # Fetch all running instances
  instances=$(gcloud compute instances list --filter="status=RUNNING" --format="value(name,zone)")

  # Loop over instances and fetch their console output
  while IFS= read -r line; do
    instance_name=$(echo $line | awk '{print $1}')
    instance_zone=$(echo $line | awk '{print $2}')

    echo "Instance Name: $instance_name"
    echo "Instance Zone: $instance_zone"

    # Fetch last 10 lines of console output
    console_output=$(gcloud compute instances get-serial-port-output $instance_name --zone $instance_zone | tail -n 10)

    echo "Console Output: "
    echo "$console_output"
    echo "-------------"
  done <<< "$instances"

}

gcloud-ssh-all(){
  # Set the command you want to run on each instance
  command_to_run="$@"  # Replace this with your command

  # Get all running instances in the project
  instances=$(gcloud compute instances list --filter="status:RUNNING" --format="value(name,zone)")

  # Loop through all instances
  while IFS= read -r instance; do
    name=$(echo $instance | cut -f1 -d' ')
    zone=$(echo $instance | cut -f2 -d' ')

    # Instance details
    echo "Instance Name: $name, Zone: $zone"

    # SSH into the instance and run the command
    # We assume that SSH keys have been set up appropriately.
    # Also note that this will fail if strict host checking is enabled and the
    # instance is not in the known hosts list.
    echo "Output of command '$command_to_run':"
    gcloud compute ssh --zone "$zone" "$name" --command "$command_to_run"
    echo "------------------------------------------------------"
  done <<< "$instances"

}