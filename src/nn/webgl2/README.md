The WebGL2Model.js imports [tfjs-core](https://github.com/tensorflow/tfjs-core), 
which is licensed under the Apache License, Version 2.0.

# Tfjs-core

Files in `./tfjs-core/dist` are built from modified tfjs-core.

## How to build

### Clone the source code
```
$ git clone https://github.com/GreyZzzzzzXh/tfjs-core.git
$ git checkout webgl2_precision
```

### Install and build
```
$ npm install
$ npm run build
```

###  Copy the built tfjs-core to webml-polyfill
Copy the generated folder `tfjs-core/dist` to `webml-polyfill/src/nn/webgl2/tfjs-core/dist`.


## What is modified in tfjs-core

To achieve a higher precision on mobile, GLSL is upgraded to ***version 300 es***.

But this causes tfjs to work only on the webgl 2.0 backend. 

See https://github.com/GreyZzzzzzXh/tfjs-core/commit/39a56bf75803ae9327bbf160189f0d08f8a06416 for more detail.
