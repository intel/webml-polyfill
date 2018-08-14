The nn_ops.js is compiled by Emscripten from https://android.googlesource.com/platform/frameworks/ml
which is licensed under the Apache License, Version 2.0.

# Build

## Prerequisites
1. Python, node.js, CMake, and Java are not provided by emsdk. Make sure you have installed these beforehand with the system package manager. 


2. [Install Emscripten](http://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).

## Steps
### Clone the source code
```
$ git clone https://github.com/huningxin/ml.git
```

### Initialize and update the submodule
```
$ cd ml/nn
$ git submodule init
$ git submodule update
```

### Create a new directory `./build`
```
$ mkdir build
$ cd build
```

### Set `CMAKE_TOOLCHAIN_FILE` for cross compilation
```
$ cmake -D CMAKE_TOOLCHAIN_FILE=/yourDownloadDir/emsdk/emscripten/yourVersion/cmake/Modules/Platform/Emscripten.cmake ..
```

### Compile the source code and generate the output file
```
$ make
```
