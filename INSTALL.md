# Dependencies

## Eigen
In a folder for 3rd party dependencies,
```bash
wget http://bitbucket.org/eigen/eigen/get/3.2.5.tar.gz
tar zxvf 3.2.5.tar.gz
cd eigen-eigen-bdd17ee3b1b3/
mkdir build && cd build
cmake ..
sudo make install
```

# Build
In your development folder,
```bash
mkdir lgmath-ws && cd $_
git clone https://github.com/utiasASRL/lgmath.git
cd lgmath && git submodule update --init --remote
```

Using [catkin](https://github.com/ros/catkin) and [catkin tools](https://github.com/catkin/catkin_tools) (recommended)
```bash
cd deps/catkin && catkin build
cd ../.. && catkin build
```

Using CMake (manual)
```bash
cd .. && mkdir -p build/catkin_optional && cd $_
cmake ../../lgmath/deps/catkin/catkin_optional && make
cd ../.. && mkdir -p build/catch && cd $_
cmake ../../lgmath/deps/catkin/catch && make
cd ../.. && mkdir -p build/lgmath && cd $_
cmake ../../lgmath && make -j4
```

# CMake Build Options

1. In your lgmath build folder (`build/lgmath`[`/lgmath`])
1. Open CMake cache editor (`ccmake .` or `cmake-gui .`)

# Install (optional)

Since the catkin build produces a catkin workspace you can overlay, and the CMake build exports packageConfig.cmake files, it is unnecessary to install lgmath except in production environments. If you are really sure you need to install, you can use the following procedure.

Using catkin tools (recommended)
```bash
cd lgmath
catkin profile add --copy-active install
catkin profile set install
catkin config --install
catkin build
```

Using CMake (manual)
```bash
cd build/lgmath
sudo make install
```

# Uninstall (Optional)

If you have installed, and would like to uninstall,

Using catkin tools (recommended)
```bash
cd lgmath && catkin clean -i
```

Using CMake (manual)
```bash
cd build/lgmath && sudo make uninstall
```
