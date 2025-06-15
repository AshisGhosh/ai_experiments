#!/usr/bin/env bash
set -euo pipefail
GIT_ROOT=$(git rev-parse --show-toplevel)
DOCKER_DIR="${GIT_ROOT}/generative_models/docker"

#–– Ensure DISPLAY is set for X11 ––
if [[ -z "${DISPLAY:-}" ]]; then
  echo "WARNING: DISPLAY not set. e.g.: export DISPLAY=:0. Setting to :0" >&2
  export DISPLAY=:0
fi

#–– Parse flags ––
BUILD=false

while getopts "b" opt; do
  case "$opt" in
    b) BUILD=true ;;
    *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# enable x11 connection
if [[ -z "${DISPLAY:-}" ]]; then
  xhost +local:docker
fi

# select container
SERVICE="ai-experiments"


# build if requested
if [[ "$BUILD" == "true" ]]; then
  export DOCKER_BUILDKIT=1
  echo "🔨 Building $SERVICE..."
  cd $DOCKER_DIR && docker compose build $SERVICE
  cd -
fi

# run container
echo "🚀 Starting $SERVICE..."
cd $DOCKER_DIR && docker compose run --rm $SERVICE
cd -

# disable x11 connection
if [[ -z "${DISPLAY:-}" ]]; then
  xhost -local:docker
fi