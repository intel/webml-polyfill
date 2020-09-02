// <<<<<<<<<<< Part A (Must Modify) -----------

// Sample url for WORKLOAD_URL: http(s)://<test-server-ip or test-server-hostname>:<port>/workload/
WORKLOAD_URL = "modify_workload_url";

// Sample url for NIGHTLY_BUILD_URL: http://<build-server-hostname>/project/webnn/nightly/
NIGHTLY_BUILD_URL = "modify_nightly_url";

// This password is required for testing on Linux platform
// to use for installation and uninstallation chromium
LINUX_PASSWORD = "modify_password";

// ----------- Part A (Must Modify) >>>>>>>>>>>


// <<<<<<<<<<< Part B -------------------------

// TARGET_BUILD_COMMIT is only for workload test
// If you want to test workload using specified commit likes "8efd473",
// please set "8efd473" to TARGET_BUILD_COMMIT, defalt TARGET_BUILD_COMMIT is "latest"
TARGET_BUILD_COMMIT = "latest";

// Customize configure TARGET_BACKEND
// Note 'DirectML' only for Windows platform
// 1) Full for Linux platform is ['WASM', 'WebGL', 'DNNL', 'clDNN', 'IE-MKLDNN', 'IE-clDNN']
// 2) Full for Linux platform is ['WASM', 'WebGL', 'DNNL', 'clDNN', 'IE-MKLDNN', 'IE-clDNN', 'DirectML']
TARGET_BACKEND = ['WASM', 'WebGL', 'DNNL', 'clDNN',
                  'IE-MKLDNN', 'IE-clDNN', 'DirectML'];

// ITERATIONS is times for running workload test, you could configure it on your purpose
// default ITERATIONS is 200
ITERATIONS = 200;


// ############# for Regression Checking ----------

// Set REGRESSION_FLAG to be true and modify DEV_CHROMIUM_PATH for Regression Test
REGRESSION_FLAG = false;

// 1) Path sample of linux '/path/chromium-browser-unstable_86.0.4209.0-1_amd64.deb'
// 2) Path sample of windows 'C:\\path\\Chrome-bin\\chrome.exe'
DEV_CHROMIUM_PATH = 'modify_chromium_path';


// ------------- for Regression Checking ##########


// current using Image Classification test with 'MobileNet v2 (TFLite)' and 'ResNet50 v2 (ONNX)'
// two models for regression checking
const REGRESSION_TEST = {
  'Image Classification': ['MobileNet v2 (TFLite)', 'ResNet50 v2 (ONNX)']
};

const NIGHTLY_BUILD_INFO = {
  "Linux": {
    "path": "linux_x64_SUCCEED",
    "suffix": "deb"
  },
  "Windows": {
    "path": "win_x64_SUCCEED",
    "suffix": "zip"
  }
};

// BACKEND_CONFIG refers to https://github.com/intel/webml-polyfill/wiki/Proposed-Chromium-Switches-for-Backends
const BACKEND_CONFIG = {
  'WASM': {
    args: ['--no-sandbox'],
    backend: 'WASM',
    prefer: 'NONE'
  },
  'WebGL': {
    args: ['--no-sandbox'],
    backend: 'WebGL',
    prefer: 'NONE'
  },
  'DNNL': {
    args: ['--no-sandbox', '--enable-features=WebML'],
    backend: 'WebNN',
    prefer: 'FAST_SINGLE_ANSWER'
  },
  'clDNN': {
    args: ['--no-sandbox', '--enable-features=WebML'],
    backend: 'WebNN',
    prefer: 'SUSTAINED_SPEED'
  },
  'IE-MKLDNN': {
    args: ['--no-sandbox', '--use-inference-engine', '--enable-features=WebML'],
    backend: 'WebNN',
    prefer: 'FAST_SINGLE_ANSWER'
  },
  'IE-clDNN': {
    args: ['--no-sandbox', '--use-inference-engine', '--enable-features=WebML'],
    backend: 'WebNN',
    prefer: 'SUSTAINED_SPEED'
  },
  'DirectML': { // Only for Windows
    args: ['--no-sandbox', '--use-dml', '--enable-features=WebML'],
    backend: 'WebNN',
    prefer: 'SUSTAINED_SPEED'
  }
};

// CATEGORY_FILTER and MODEL_FILTER refer to https://github.com/intel/webml-polyfill/wiki/WebML-Examples-Results-on-Different-Backends-and-Platforms
// skip test such category by listed backend
const CATEGORY_FILTER = {
  'Semantic Segmentation': ["DNNL", "IE-clDNN", "IE-MKLDNN"],
  'Super Resolution': ["clDNN", "DNNL", "DirectML", "IE-clDNN", "IE-MKLDNN"],
  'Emotion Analysis': ["clDNN", "DNNL", "DirectML", "IE-clDNN", "IE-MKLDNN"],
  'Facial Landmark Detection': ["clDNN", "IE-clDNN", "IE-MKLDNN"]
};

// skip test such model by backend
const MODEL_FILTER = {
  "WASM": [
    'MobileNet v1 Quant (Caffe2)',
    'Deeplab 224 (Tensorflow)',
    'Deeplab 257 (Tensorflow)',
    'Deeplab 321 (Tensorflow)',
    'Deeplab 513 (Tensorflow)'
  ],
  "WebGL": [
    'MobileNet v1 Quant (TFLite)',
    'MobileNet v2 Quant (TFLite)',
    'Inception v3 Quant (TFLite)',
    'Inception v4 Quant (TFLite)',
    'MobileNet v1 Quant (Caffe2)',
    'SSD MobileNet v1 Quant (TFLite)',
    'SSD MobileNet v2 Quant (TFLite)',
    'Deeplab 224 (Tensorflow)',
    'Deeplab 257 (Tensorflow)',
    'Deeplab 321 (Tensorflow)',
    'Deeplab 513 (Tensorflow)'
  ],
  "clDNN": [
    'MobileNet v1 Quant (TFLite)',
    'MobileNet v2 Quant (TFLite)',
    'Inception v3 Quant (TFLite)',
    'Inception v4 Quant (TFLite)',
    'MobileNet v1 Quant (Caffe2)',
    'SSD MobileNet v1 Quant (TFLite)',
    'SSD MobileNet v2 Quant (TFLite)',
    'Tiny Yolo v2 COCO (TFLite)',
    'Tiny Yolo v2 VOC (TFLite)',
    'Deeplab 224 (Tensorflow)',
    'Deeplab 257 (Tensorflow)',
    'Deeplab 321 (Tensorflow)',
    'Deeplab 513 (Tensorflow)'
  ],
  "DNNL": [
    'MobileNet v1 Quant (TFLite)',
    'MobileNet v2 Quant (TFLite)',
    'Inception v3 Quant (TFLite)',
    'Inception v4 Quant (TFLite)',
    'Inception v2 (ONNX)',
    'DenseNet 121 (ONNX)',
    'SqueezeNet (OpenVino)',
    'DenseNet 121 (OpenVino)',
    'SSD MobileNet v1 Quant (TFLite)',
    'SSD MobileNet v2 Quant (TFLite)',
    'Tiny Yolo v2 COCO (TFLite)',
    'Tiny Yolo v2 VOC (TFLite)',
    'Tiny Yolo v2 Face (TFlite)'
  ],
  "DirectML": [
    'MobileNet v1 Quant (TFLite)',
    'MobileNet v2 Quant (TFLite)',
    'Inception v3 Quant (TFLite)',
    'Inception v4 Quant (TFLite)',
    'DenseNet 121 (ONNX)',
    'SqueezeNet (OpenVino)',
    'DenseNet 121 (OpenVino)',
    'MobileNet v1 Quant (Caffe2)',
    'SSD MobileNet v1 Quant (TFLite)',
    'SSD MobileNet v2 Quant (TFLite)',
    'Deeplab 257 (TFLite)',
    'Deeplab 257 Atrous (TFLite)',
    'Deeplab 321 (TFLite)',
    'Deeplab 321 Atrous (TFLite)',
    'Deeplab 513 (TFLite)',
    'Deeplab 513 Atrous (TFLite)',
    'Deeplab 257 Atrous (OpenVINO)',
    'Deeplab 321 Atrous (OpenVINO)',
    'Deeplab 513 Atrous (OpenVINO)',
    'Deeplab 224 (Tensorflow)',
    'Deeplab 257 (Tensorflow)',
    'Deeplab 321 (Tensorflow)',
    'Deeplab 513 (Tensorflow)',
    'Tiny Yolo v2 COCO (TFLite)',
    'Tiny Yolo v2 VOC (TFLite)',
    'Tiny Yolo v2 Face (TFlite)'
  ],
  "IE-clDNN": [
    'MobileNet v1 Quant (TFLite)',
    'MobileNet v2 Quant (TFLite)',
    'Inception v3 Quant (TFLite)',
    'Inception v4 (TFLite)',
    'Inception v4 Quant (TFLite)',
    'Inception ResNet v2 (TFLite)',
    'Inception v2 (ONNX)',
    'DenseNet 121 (ONNX)',
    'ResNet50 v1 (ONNX)',
    'ResNet50 v2 (ONNX)',
    'Inception v2 (ONNX)',
    'DenseNet 121 (ONNX)',
    'ResNet50 v1 (OpenVino)',
    'DenseNet 121 (OpenVino)',
    'Inception v2 (OpenVino)',
    'Inception v4 (OpenVino)',
    'MobileNet v1 Quant (Caffe2)',
    'SSD MobileNet v1 Quant (TFLite)',
    'SSD MobileNet v2 Quant (TFLite)',
    'Tiny Yolo v2 COCO (TFLite)',
    'Tiny Yolo v2 VOC (TFLite)',
    'Tiny Yolo v2 Face (TFlite)'
  ],
  "IE-MKLDNN": [
    'MobileNet v1 Quant (TFLite)',
    'MobileNet v2 Quant (TFLite)',
    'Inception v3 Quant (TFLite)',
    'Inception v4 Quant (TFLite)',
    'Inception ResNet v2 (TFLite)',
    'Inception v2 (ONNX)',
    'DenseNet 121 (ONNX)',
    'Inception v2 (ONNX)',
    'DenseNet 121 (ONNX)',
    'DenseNet 121 (OpenVino)',
    'MobileNet v1 Quant (Caffe2)',
    'SSD MobileNet v1 Quant (TFLite)',
    'SSD MobileNet v2 Quant (TFLite)',
    'Tiny Yolo v2 COCO (TFLite)',
    'Tiny Yolo v2 VOC (TFLite)',
    'Tiny Yolo v2 Face (TFlite)'
  ]
};

// ----------- Part B >>>>>>>>>>>>>>>>>>>>>>>>>

module.exports = {
  WORKLOAD_URL: WORKLOAD_URL,
  NIGHTLY_BUILD_URL: NIGHTLY_BUILD_URL,
  LINUX_PASSWORD: LINUX_PASSWORD,
  TARGET_BUILD_COMMIT: TARGET_BUILD_COMMIT,
  NIGHTLY_BUILD_INFO: NIGHTLY_BUILD_INFO,
  TARGET_BACKEND: TARGET_BACKEND,
  ITERATIONS: ITERATIONS,
  BACKEND_CONFIG: BACKEND_CONFIG,
  CATEGORY_FILTER: CATEGORY_FILTER,
  MODEL_FILTER: MODEL_FILTER,
  REGRESSION_FLAG: REGRESSION_FLAG,
  REGRESSION_TEST: REGRESSION_TEST,
  DEV_CHROMIUM_PATH: DEV_CHROMIUM_PATH,
};
