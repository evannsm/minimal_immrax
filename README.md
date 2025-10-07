## Minimal_Immrax
This is a clean, minimal version of immrax without NN functionality, linrax, or any other bells and whistles beyond the basic MM reachability functionality. It is also a customized version that fixes some issues the official immrax repo has with working within the RTA-GPR pipeline.

This is a lightweight package that could also be easier to use for raspberry pi onboard computation.



## Immrax Itself
`immrax` is a tool for interval analysis and mixed monotone reachability analysis in JAX.

Inclusion function transformations are composable with existing JAX transformations, allowing the use of Automatic Differentiation to learn relationships between inputs and outputs, as well as parallelization and GPU capabilities for quick, accurate reachable set estimation.

For more information, please see the full [documentation](https://immrax.readthedocs.io).

## Installation
If cloning the Github repository, you can `pip install` it with the following commands:
```shell
git clone https://github.com/gtfactslab/immrax.git
cd immrax
pip install . # can use -e for editable install
```

## **Usage with ROS 2**

A companion repository, [ROS2 Minimal Immrax](https://github.com/evannsm/ROS2MinimalImmrax), provides a version designed to run easily **within a ROS 2 package** *without requiring installation*.

If you prefer to install this package system-wide or use it outside ROS 2, consider one of the following approaches:

### **1. System-Wide pip Install**
While possible, this approach is **not recommended**, as system-level installations can cause dependency conflicts.  
However, because ROS 2 does not integrate cleanly with virtual environments, some users choose this route for simplicity.

### **2. Docker Container**
Containerization is a robust way to manage dependencies and ensure reproducibility.  
You can base your container on the [official ROS 2 Docker images](https://hub.docker.com/_/ros) and install `immrax` within your custom `Dockerfile`.

### **3. RoboStack (Mamba Environments)**
[RoboStack](https://robostack.github.io/robostack.github.io/) provides ROS 2-compatible Conda/Mamba environments, allowing you to manage packages like `immrax` cleanly alongside ROS 2 dependencies.
