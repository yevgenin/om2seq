#!/bin/bash

set -o allexport
source .env
source .env.local
set +o allexport

pyrun(){
  python src/$PROJECT_NAME/$@
}

$@