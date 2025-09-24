# Clean Immrax Package for Basic Use and Clean Dependencies

# 1. For general mamba environments with immrax:
- First create your mamba environment for this
```bash
mamba create -y -n immrax_env python=3.12 -c conda-forge
```
- Secondly, 
```bash
git clone git@github.com:evannsm/minimal_immrax.git && cd minimal_immrax
pip install . # editable installs are nice but for some reason ROS2 has trouble finding the package that way. i'm sure there's some way to work around this but this works for now
mamba install -c conda-forge matplotlib-base=3.10.6 matplotlib-inline=0.1.7
pip uninstall -y matplotlib || true
```

# 2. For Robostack environments with immrax: 
- First create your robostack mamba environment for this
```bash
mamba create -n <ros_env_name>
mamba activate <ros_env_name>
```

```bash
# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

conda config --env --add channels robostack-jazzy
mamba install ros-jazzy-desktop

mamba deactivate # Deactivate and reactivate the environment to initialize the configured ROS environment
mamba activate <ros_env_name>

mamba install compilers cmake pkg-config make ninja colcon-common-extensions rosdep
```

- Secondly, 
```bash
git clone git@github.com:evannsm/minimal_immrax.git && cd minimal_immrax
pip install . # editable installs are nice but for some reason ROS2 has trouble finding the package that way. i'm sure there's some way to work around this but this works for now
mamba install -c conda-forge matplotlib-base=3.10.6 matplotlib-inline=0.1.7
pip uninstall -y matplotlib || true
```
