#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash
source /opt/wm-venv/bin/activate
exec "$@"
