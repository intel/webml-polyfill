# Deprecated

W3C Launched the Web Machine Learning Working Group in April 2021, the Web Neural Network Polyfill project has been moved to [https://github.com/webmachinelearning/webnn-polyfill](https://github.com/webmachinelearning/webnn-polyfill). You can also get latest updates for the Web Neural Network API and other relevant projects in [github.com/webmachinelearning](https://github.com/webmachinelearning).

# Web Machine Learning

## Web Neural Network (WebNN) API polyfill and examples

A polyfill for [Web Neural Network (WebNN) API](https://webmachinelearning.github.io/webnn/) with computer vision and natural language processing examples.

The [Web Neural Network (WebNN) API](https://webmachinelearning.github.io/webnn/) is a dedicated low-level API for neural network inference hardware acceleration. It is worked on in the W3C [Machine Learning for the Web Community Group](https://www.w3.org/community/webmachinelearning/).

## Project Build Status

MacOS | Linux | Windows
-------- | -------- | --------
[![Build Status](https://api.travis-ci.com/intel/webml-polyfill.svg?branch=master)](https://travis-ci.com/intel/webml-polyfill) | [![CircleCI](https://circleci.com/gh/intel/webml-polyfill/tree/master.svg?style=svg)](https://circleci.com/gh/intel/webml-polyfill/tree/master) | [![Build status](https://ci.appveyor.com/api/projects/status/6xjudmjja1mcyo1m/branch/master?svg=true)](https://ci.appveyor.com/project/ibelem/webml-polyfill-egsl9/branch/master)

## Examples

* [WebNN API Examples](https://intel.github.io/webml-polyfill/examples/)
* [WebNN Meeting (Intelligent Collaboration)](https://github.com/intel/webml-polyfill/tree/master/examples/webnnmeeting)

<img src="./examples/static/img/qr.png" width="160" height="160" alt="WebNN API Examples QR Code">

### Supported Backends

* Polyfill
  * WASM: TensorFlow.js WebAssembly backend builds on top of the XNNPACK library
  * WebGL: TensorFlow.js GPU accelerated WebGL backend
  * WebGPU: WIP
* WebNN: Web Neural Network (WebNN) API

### Run example with hardware accelerated WebNN backend

If you are interested, please refer to [WebNN Chromium build repo](https://github.com/otcshare/chromium-src) and WIKI:

* How to build WebNN Chromium on [Windows](https://github.com/intel/webml-polyfill/wiki/How-to-build-chromium-on-Windows), [Linux](https://github.com/intel/webml-polyfill/wiki/How-to-build-chromium-on-Linux), [macOS](https://github.com/intel/webml-polyfill/wiki/How-to-build-chromium-on-macOS), [ChromeOS](https://github.com/intel/webml-polyfill/wiki/How-to-build-chromium-on-ChromeOS) and [Android](https://github.com/intel/webml-polyfill/wiki/How-to-build-chromium-for-Android)
* [How to run Chromium builds with WebNN API](https://github.com/intel/webml-polyfill/wiki/How-to-Run-Chromium-builds-with-WebNN-API)

### Benchmarks

* [Web AI Workload](https://intel.github.io/webml-polyfill/workload/) Use this tool to collect the performance-related metrics (inference time, etc) of various models and kernels on your local device with Wasm, WebGL, or WebNN backends. The [Web AI Workload](https://intel.github.io/webml-polyfill/workload/) also supports to measure the OpenCV.js DNN performance with Wasm, Wasm Threads and Wasm SIMD.
* [OpenCV.js Performance Test](https://intel.github.io/webml-polyfill/workload/opencv_threshold) Use this tool to collect the OpenCV.js performance for image processing with Wasm, Wasm Threads and Wasm SIMD.

## Building & Testing

### Install

```sh
$ npm install
```

### Start

```sh
$ npm start

# Start an HTTPS server
$ HTTPS=true npm start
```

### Build

```sh
$ npm run build

# Production build
$ NODE_ENV=production npm run build

# WASM backend build:
$ npm run build-wasm
```

### Test

```sh
$ npm start
```

Open browser and navigate to http://localhost:8080/test

### Watch

```sh
$ npm run watch
```

## License

This project is following [Apache License Version 2.0](./LICENSE_APACHE2).

> Documents in [test/wpt/resources](./test/wpt/resources) are licensed by the [W3C 3-clause BSD License](./test/wpt/resources/LICENSE).
