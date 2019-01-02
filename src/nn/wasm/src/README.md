The source code can be compiled to `nn_ops.js` by Emscripten, which is used for webml-polyfill WASM backend.

# WASM Build

## Prerequisites
1. Python, node.js, CMake, and Java are not provided by emsdk. Make sure you have installed these beforehand with the system package manager. 


2. [Install Emscripten](http://kripken.github.io/emscripten-site/docs/getting_started/downloads.html).

## Steps
### Initialize and update the submodule
```
$ cd webml-polyfill/
$ git submodule update --init --recursive
```

### Download tensorflow dependencies
```
$ cd src/nn/wasm/src/external/tensorflow/
$ ./tensorflow/contrib/makefile/download_dependencies.sh
```

### Create a new directory `./build`
```
$ cd ../../
$ mkdir build
$ cd build
```

### Set `CMAKE_TOOLCHAIN_FILE` for cross compilation
```
$ cmake -D CMAKE_TOOLCHAIN_FILE=/yourDownloadDir/emsdk/emscripten/yourVersion/cmake/Modules/Platform/Emscripten.cmake ..
```

### Compile the source code and generate the output file nn_ops.js
```
$ make
```
