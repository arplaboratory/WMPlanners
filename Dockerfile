# ROS 2 Jazzy provides the Python 3.12 runtime required by wm_planner_ros.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/wm-venv \
    PATH=/opt/wm-venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3-pip python3-venv python3-dev python3-colcon-common-extensions \
      libgl1 libglib2.0-0 libgomp1 \
      ros-jazzy-message-filters ros-jazzy-tf2-ros \
      ros-jazzy-tf2-geometry-msgs ros-jazzy-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/src/wm_planner
COPY . .

# The backbone loaders use the local submodule paths at runtime.
RUN test -f external/dino/setup.py && test -f external/dinov3/setup.py \
    && test -f external/vjepa2/setup.py
RUN python3 -m venv --system-site-packages "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -e . \
    && python -m pip install --no-cache-dir --no-deps -e external/dino \
       -e external/dinov3 -e external/vjepa2 \
    && python -m pip install --no-cache-dir fvcore iopath torchmetrics peft fire python-box termcolor

WORKDIR /workspace
RUN /bin/bash -c 'source /opt/ros/jazzy/setup.bash && colcon build --base-paths src/wm_planner/ros2 --packages-select wm_planner_ros'

COPY docker/entrypoint.sh /usr/local/bin/wm-planner-entrypoint
RUN chmod +x /usr/local/bin/wm-planner-entrypoint
ENTRYPOINT ["/usr/local/bin/wm-planner-entrypoint"]
CMD ["bash"]
