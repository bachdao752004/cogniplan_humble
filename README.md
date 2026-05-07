# CogniPlan_humble (ROS2)

ROS2 Humble planner package for CogniPlan, with two tasks:

- `exploration`: learned exploration policy from `expl_model`
- `navigation`: learned navigation policy from `nav_model`

## 1) Requirements

- Ubuntu 22.04 + ROS2 Humble
- `octomap_server`
- Python environment (recommended: conda `ros2-torch`)

Install ROS dependency:

```bash
sudo apt-get install ros-humble-octomap-server
```

Create/update conda env (CPU inference):

```bash
conda create -n ros2-torch python=3.10.12 -y
conda activate ros2-torch
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
python -m pip install numpy==1.26.4 scipy scikit-image==0.22.0
python -m pip install opencv-python-headless pillow imageio pyyaml rospkg
python -m pip install -U colcon-common-extensions distlib
```

Recommended package checklist in `ros2-torch`:

- Deep learning: `torch`, `torchvision`, `torchaudio` (CPU wheels)
- Numeric/scientific: `numpy==1.26.4`, `scipy`, `scikit-image==0.22.0`
- Vision/io: `opencv-python-headless`, `pillow`, `imageio`
- ROS/python utils: `pyyaml`, `rospkg`, `colcon-common-extensions`, `distlib`

## 2) Environment rules (important)

To avoid ROS2 + conda + user-site conflicts, run these in **every terminal** before build/launch:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

## 3) Build

```bash
conda activate ros2-torch
unset PYTHONPATH
export PYTHONNOUSERSITE=1

cd /media/bach/bach/nav_expl/CogniPlan_humble
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-select rl_planner
source install/setup.bash
```

Daily quick start (copy/paste):

```bash
conda activate ros2-torch
unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd /media/bach/bach/nav_expl/CogniPlan_humble
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-select rl_planner
source install/setup.bash
```

## 4) Run (4 terminals, recommended)

### Terminal A - simulator

```bash
conda activate ros2-torch
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash

cd /path/to/CogniPlan_humble
source autonomous_exploration_development_environment/devel/setup.bash
source install/setup.bash
ros2 launch vehicle_simulator system_indoor.launch
```

Other simulator environments:

```bash
# forest
ros2 launch vehicle_simulator system_forest.launch

# tunnel
ros2 launch vehicle_simulator system_tunnel.launch

# garage
ros2 launch vehicle_simulator system_garage.launch

# campus
ros2 launch vehicle_simulator system_campus.launch
```

Use exactly one simulator launch at a time (do not run multiple `system_*.launch` files concurrently).

### Terminal B - planner

```bash
conda activate ros2-torch
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash

cd /path/to/CogniPlan_humble
source install/setup.bash
```

Run exploration:

```bash
ros2 launch rl_planner rl_planner.launch.py task:=exploration
```

Run navigation:

```bash
ros2 launch rl_planner rl_planner.launch.py task:=navigation
```

For navigation, set destination in RViz using `2D Goal Pose` (topic `/goal_pose`).

### Terminal C - publish/monitor topics

```bash
conda activate ros2-torch
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash

cd /path/to/CogniPlan_humble
source install/setup.bash
```

Publish a goal from command line:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: 'map'},
  pose: {position: {x: 5.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}
}"
```

Monitor key topics:

```bash
ros2 topic hz /way_point
ros2 topic hz /projected_map
ros2 topic echo /runtime
```

### Terminal D - rosbag recording

```bash
conda activate ros2-torch
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash

cd /path/to/CogniPlan_humble
source install/setup.bash
mkdir -p bags
ros2 bag record -o bags/run_$(date +%Y%m%d_%H%M%S) \
/projected_map /way_point /runtime /state_estimation /sensor_scan /tf /tf_static /goal_pose /nav_goal_marker /nav_path
```

## 5) Models

Runtime model directories:

- `src/rl_planner/rl_planner/expl_model` (exploration)
- `src/rl_planner/rl_planner/nav_model` (navigation)

Replace `checkpoint.pth` (and related files if needed) to use your own trained models.

## 6) Logging

ROS2 launch logs are saved under:

```bash
~/.ros/log/
```

## 7) Troubleshooting

- `ModuleNotFoundError: No module named 'torch'`
  - Install torch in active env:
    - `python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
- `numpy/scikit-image binary incompatibility`
  - Reinstall in env:
    - `python -m pip install --upgrade --force-reinstall "numpy==1.26.4" "scikit-image==0.22.0"`
  - Ensure imports are not from `~/.local/...`
- RViz `No map received` on `/overall_map`
  - Check topic type. If `/overall_map` is `PointCloud2`, add it as `PointCloud2` display, not `Map`.

