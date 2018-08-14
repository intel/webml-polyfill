The nn_ops.js is compiled by Emscripten from https://android.googlesource.com/platform/frameworks/ml
which is licensed under the Apache License, Version 2.0.



Build steps:

1. install Emscripten: (Reference: http://kripken.github.io/emscripten-site/docs/getting_started/downloads.html) 
    --# Get the emsdk repo
        git clone https://github.com/juj/emsdk.git

    --# Enter that directory
        cd emsdk

    --# Fetch the latest registry of available tools. 
        ./emsdk update

    --# Download and install the latest SDK tools.  
        ./emsdk install latest

    --# Make the "latest" SDK "active" for the current user. (writes ~/.emscripten file)
        ./emsdk activate latest

    --# Activate PATH and other environment variables in the current terminal
        source ./emsdk_env.sh

2. Python, node.js, CMake, and Java are not provided by emsdk.Make sure you have installed these beforehand with the system package manager:

    --# Install Python
        sudo apt-get install python2.7

    --# Install node.js
        sudo apt-get install nodejs

    --# Install CMake (optional, only needed for tests and building Binaryen)
        sudo apt-get install cmake

    --# Install Java (optional, only needed for Closure Compiler minification)
        sudo apt-get install default-jre

    Note: 
    (1) You need Python 2.7.12 or newer because older versions may not work due to a GitHub change with SSL(https://github.com/kripken/emscripten/issues/6275)
    (2) Your system may provide Node.js as node instead of nodejs. In that case, you may need to also update the NODE_JS attribute of your ~/.emscripten file.

3. git clone https://github.com/huningxin/ml.git

4. cd ml/nn
   git submodule init
   git submodule update

5. mkdir build
   cd build

6. cmake -D CMAKE_TOOLCHAIN_FILE=/yourDownloadDir/emsdk/emscripten/1.38.11/cmake/Modules/Platform/Emscripten.cmake .. (there are two dots)

7. make