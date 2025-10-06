# lgmath

lgmath is a C++ library for handling geometry in state estimation problems in robotics.
It is used to store, manipulate, and apply three-dimensional rotations and transformations and their associated uncertainties.

There are no minimal, constraint-free, singularity-free representations for these quantities, so lgmath exploits two different representations for the nominal and noisy parts of the uncertain random variable.

- Nominal rotations and transformations are represented using their composable, singularity-free _matrix Lie groups_, **SO(2)**/**SO(3)** and **SE(2)**/**SE(3)**.
- Their uncertainties are represented as multiplicative perturbations on the minimal, constraint-free vectorspaces of their _Lie algebras_, **so(2)**/**so(3)** and **se(2)**/**se(3)**.

This library uses concepts and mathematics described in Timothy D. Barfoot's book [State Estimation for Robotics](asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf).
It is used for robotics research at the Autonomous Space Robotics Lab; most notably in the STEAM Engine, a library for Simultaneous Trajectory Estimation and Mapping.

## Installation

### Dependencies

- Compiler with C++17 support
- CMake (>=3.16)
- Eigen (>=3.3.7)
- (Optional) ROS2 Foxy or later (colcon+ament_cmake)

### Install c++ compiler and cmake

```bash
sudo apt -q -y install build-essential cmake
```

### Install Eigen (>=3.3.7)

```bash
# using APT
sudo apt -q -y install libeigen3-dev

# OR from source
WORKSPACE=~/workspace  # choose your own workspace directory
mkdir -p ${WORKSPACE}/eigen && cd $_
git clone https://gitlab.com/libeigen/eigen.git . && git checkout 3.3.7
mkdir build && cd $_
cmake .. && make install # default install location is /usr/local/
```

- Note: if installed from source to a custom location then make sure `cmake` can find it. Also ensure that ros has not been sourced in the terminal used for compilation otherwise it will be compiled as an `ament_cmake` package.

### Build and install lgmath using `cmake`

```bash
WORKSPACE=~/workspace  # choose your own workspace directory
# clone
mkdir -p ${WORKSPACE}/lgmath && cd $_
git clone https://github.com/utiasASRL/lgmath.git .
# build and install
mkdir -p build && cd $_
cmake ..  # optionally include -DBUILD_TESTING=ON to build tests
cmake --build .
cmake --install . # (optional) install, default location is /usr/local/
make doc  # (optional) generate documentation in ./doc
```

Note: `lgmathConfig.cmake` will be generated in both `build/` and `<install prefix>/lib/cmake/lgmath/` to be included in other projects.

### Build and install lgmath using `ROS2(colcon+ament_cmake)`

CMake will automatically determine if you are building as part of a 
```bash
colcon build
```
and use `ament_cmake` for you. This is achieved checking if ROS is present in your terminal and if the install location is not the default `/usr/local` which is true of ROS packages.


## [License](./LICENSE)
