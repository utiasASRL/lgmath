# Dependencies

## Eigen
In a folder for 3rd party dependencies,
```
wget http://bitbucket.org/eigen/eigen/get/3.2.5.tar.gz
tar zxvf 3.2.5.tar.gz
cd eigen-eigen-bdd17ee3b1b3/
mkdir build && cd build
cmake ..
sudo make install
```

# Install
In your development folder,
```
mkdir lgmath && cd lgmath
git clone https://github.com/utiasASRL/lgmath.git src
mkdir build && cd build
cmake ../src
make -j4
sudo make install
```

# Enable Unit Tests 
(Optional)

1. Open CMake App
1. Enable TESTS_ON
1. cd build && make

# Uninstall
(Optional)

```
cd build
sudo make uninstall
```
