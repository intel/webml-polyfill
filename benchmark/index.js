'use strict';
const tfliteModelArray = [
  "mobilenet_v1_tflite",
  "mobilenet_v2_tflite",
  "inception_v3_tflite",
  "inception_v4_tflite",
  "squeezenet_tflite",
  "inception_resnet_v2_tflite"];

const ssdModelArray = [
  "ssd_mobilenet_v1_tflite",
  "ssd_mobilenet_v2_tflite",
  "ssdlite_mobilenet_v2_tflite"];

const onnxModelArray = [
  "squeezenet_onnx",
  "mobilenet_v2_onnx",
  "resnet_v1_onnx",
  "resnet_v2_onnx",
  "inception_v2_onnx",
  "densenet_onnx"];

let supportedModels = [];
supportedModels = supportedModels.concat(imageClassificationModels, objectDetectionModels, humanPoseEstimationModels);

let imageElement = null;
let inputElement = null;
let pickBtnEelement = null;
let canvasElement = null;
let poseCanvas = null;
let bkPoseImageSrc = null;
let pnConfigDic = null;

let preferDivElement = document.getElementById('preferDiv');
let preferSelectElement = document.getElementById('preferSelect');

function getModelDicItem(modelFormatName) {
  for (let model of supportedModels) {
    if (model.modelFormatName === modelFormatName) {
      return model;
    }
  }
}

function getPreferString(backend) {
  let prefer;
  let backendLC = backend.toLowerCase();

  if (backendLC === 'wasm') {
    prefer = 'fast';
  } else if (backendLC === 'webgl') {
    prefer = 'sustained';
  } else if (backendLC === 'webml') {
    prefer = preferSelectElement.options[preferSelectElement.selectedIndex].value;
  }

  return prefer;
}

const util = new Utils();
class Logger {
  constructor($dom) {
    this.$dom = $dom;
    this.indent = 0;
  }
  log(message) {
    console.log(message);
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + message}`;
  }
  error(err) {
    console.error(err);
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + err.message}`;
  }
  group(name) {
    console.group(name);
    this.log('');
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + name}`;
    this.indent++;
  }
  groupEnd() {
    console.groupEnd();
    this.indent--;
  }
}
class Benchmark {
  constructor() {
    this.summary = null;
  }
  async runAsync(configuration) {
    this.configuration = configuration;
    await this.setupAsync();
    let results = await this.executeAsync();
    await this.finalizeAsync();
    return {"computeResults": this.summarize(results.computeResults),
            "decodeResults": this.summarize(results.decodeResults)};
  }
  /**
   * Setup model
   * @returns {Promise<void>}
   */
  async setupAsync() {
    throw Error("Not Implemented");
  }

  async loadImage(canvas, width, height) {
    let ctx = canvas.getContext('2d');
    let image = new Image();
    let promise = new Promise((resolve, reject) => {
      image.crossOrigin = '';
      image.onload = () => {
        canvas.width = width;
        canvas.height = height;
        canvas.setAttribute("width", width);
        canvas.setAttribute("height", height);
        ctx.drawImage(image, 0, 0, width, height);
        resolve(image);
      };
    });
    image.src = imageElement.src;
    return promise;
  }

  async executeAsync() {
    let computeResults = [];
    let decodeResults = [];
    let modelName = this.configuration.modelName;
    if (tfliteModelArray.indexOf(modelName) !== -1 || onnxModelArray.indexOf(modelName) !== -1) {
      for (let i = 0; i < this.configuration.iteration; i++) {
        this.onExecuteSingle(i);
        await new Promise(resolve => requestAnimationFrame(resolve));
        let tStart = performance.now();
        await this.executeSingleAsync();
        let elapsedTime = performance.now() - tStart;
        this.printPredictResult();
        computeResults.push(elapsedTime);
      }
    } else if (modelName === 'posenet') {
      let singlePose = null;
      for (let i = 0; i < this.configuration.iteration; i++) {
        this.onExecuteSingle(i);
        await new Promise(resolve => requestAnimationFrame(resolve));
        let tStart = performance.now();
        await this.executeSingleAsyncPN();
        let elapsedTime = performance.now() - tStart;
        computeResults.push(elapsedTime);
        let dstart = performance.now();
        singlePose = decodeSinglepose(sigmoid(this.heatmapTensor), this.offsetTensor,
                                      toHeatmapsize(this.scaleInputSize, this.outputStride),
                                      this.outputStride);
        let decodeTime = performance.now() - dstart;
        console.log("Decode time:" + decodeTime);
        decodeResults.push(decodeTime);
      }
      // draw canvas by last result
      await this.loadImage(poseCanvas, imageElement.width, imageElement.height);
      let ctx = poseCanvas.getContext('2d');
      let scaleX = poseCanvas.width / this.scaleWidth;
      let scaleY = poseCanvas.height / this.scaleHeight;
      singlePose.forEach((pose) => {
        if (pose.score >= this.minScore) {
          drawKeypoints(pose.keypoints, this.minScore, ctx, scaleX, scaleY);
          drawSkeleton(pose.keypoints, this.minScore, ctx, scaleX, scaleY);
        }
      });
      bkPoseImageSrc = imageElement.src;
      imageElement.src = poseCanvas.toDataURL();
    } else if (ssdModelArray.indexOf(modelName) !== -1) {
      for (let i = 0; i < this.configuration.iteration; i++) {
        this.onExecuteSingle(i);
        await new Promise(resolve => requestAnimationFrame(resolve));
        let tStart = performance.now();
        await this.executeSingleAsyncSSDMN();
        let elapsedTime = performance.now() - tStart;
        computeResults.push(elapsedTime);
        let dstart = performance.now();
        decodeOutputBoxTensor({}, this.outputBoxTensor, this.anchors);
        let decodeTime = performance.now() - dstart;
        console.log("Decode time:" + decodeTime);
        decodeResults.push(decodeTime);
      }
      let [totalDetections, boxesList, scoresList, classesList] = NMS({}, this.outputBoxTensor, this.outputClassScoresTensor);
      poseCanvas.setAttribute("width", imageElement.width);
      poseCanvas.setAttribute("height", imageElement.height);
      visualize(poseCanvas, totalDetections, imageElement, boxesList, scoresList, classesList, this.labels);
      bkPoseImageSrc = imageElement.src;
      imageElement.src = poseCanvas.toDataURL();
    }
    return {"computeResults": computeResults, "decodeResults": decodeResults};
  }
  /**
   * Execute model
   * @returns {Promise<void>}
   */
  async executeSingleAsync() {
    throw Error('Not Implemented');
  }
  /**
   * Execute PoseNet model
   * @returns {Promise<void>}
   */
  async executeSingleAsyncPN() {
    throw Error('Not Implemented');
  }
  /**
   * Execute SSD MobileNet model
   * @returns {Promise<void>}
   */
  async executeSingleAsyncSSDMN() {
    throw Error('Not Implemented');
  }
  /**
   * Finalize
   * @returns {Promise<void>}
   */
  async finalizeAsync() {}
  summarize(results) {
    if (results.length !== 0) {
      results.shift(); // remove first run, which is regarded as "warming up" execution
      let d = results.reduce((d, v) => {
        d.sum += v;
        d.sum2 += v * v;
        return d;
      }, {
        sum: 0,
        sum2: 0
      });
      let mean = d.sum / results.length;
      let std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
      return {
        configuration: this.configuration,
        mean: mean,
        std: std,
        results: results
      };
    } else {
      return null;
    }
  }
  onExecuteSingle(iteration) {}
}
class WebMLJSBenchmark extends Benchmark {
  constructor() {
    super(...arguments);
    this.inputTensor = null;
    // outputTensor only for mobilenet and squeezenet
    this.outputTensor = null;

    // only for posenet
    this.modelVersion = null;
    this.outputStride = null;
    this.scaleFactor = null;
    this.minScore = null;
    this.scaleWidth = null;
    this.scaleHeight = null;
    this.scaleInputSize = null;
    this.heatmapTensor = null;
    this.offsetTensor  = null;

    //only for ssd mobilenet
    this.outputBoxTensor = null;
    this.outputClassScoresTensor = null;
    this.anchors = null;

    this.model = null;
    this.labels = null;

  }
  async loadModelAndLabels(model) {
    let url = '../examples/util/';
    let arrayBuffer = await this.loadUrl(url + model.modelFile, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(url + model.labelsFile);
    return {
      bytes: bytes,
      text: text
    };
  }
  async loadUrl(url, binary) {
    return new Promise((resolve, reject) => {
      let request = new XMLHttpRequest();
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function (ev) {
        if (request.readyState === 4) {
          if (request.status === 200) {
            resolve(request.response);
          } else {
            reject(new Error('Failed to load ' + url + ' status: ' + request.status));
          }
        }
      };
      request.send();
    });
  }
  async setInputOutput() {
    const configModelName = this.configuration.modelName;
    const currentModel = getModelDicItem(configModelName);
    let width = currentModel.inputSize[1];
    let height = currentModel.inputSize[0];
    const channels = currentModel.inputSize[2];
    const preOptions = currentModel.preOptions || {};
    const mean = preOptions.mean || [0, 0, 0, 0];
    const std  = preOptions.std  || [1, 1, 1, 1];
    const norm = preOptions.norm || false;
    const channelScheme = preOptions.channelScheme || 'RGB';
    const imageChannels = 4; // RGBA
    let drawContent;

    if (tfliteModelArray.indexOf(configModelName) !== -1 || onnxModelArray.indexOf(configModelName) !== -1) {
      this.inputTensor = new Float32Array(currentModel.inputSize.reduce((a, b) => a * b));
      this.outputTensor = new Float32Array(currentModel.outputSize);
      drawContent = imageElement;
    } else if (ssdModelArray.indexOf(configModelName) !== -1) {
      if (bkPoseImageSrc !== null) {
        // reset for rerun with same image
        imageElement.src = bkPoseImageSrc;
      }
      this.inputTensor = new Float32Array(currentModel.inputSize.reduce((a, b) => a * b));
      this.outputTensor = [];
      this.outputBoxTensor = new Float32Array(currentModel.num_boxes * currentModel.box_size);
      this.outputClassScoresTensor = new Float32Array(currentModel.num_boxes * currentModel.num_classes);
      this.prepareoutputTensor(this.outputBoxTensor, this.outputClassScoresTensor);
      this.anchors = generateAnchors({});
      drawContent = imageElement;
    } else if (configModelName === 'posenet') {
      if (bkPoseImageSrc !== null) {
        // reset for rerun with same image
        imageElement.src = bkPoseImageSrc;
      }
      if (pnConfigDic === null) {
        // Read modelVersion outputStride scaleFactor minScore from json file
        let posenetConfigURL = './posenetConfig.json';
        let pnConfigText = await this.loadUrl(posenetConfigURL);
        pnConfigDic = JSON.parse(pnConfigText);
      }
      this.modelVersion = Number(pnConfigDic.modelVersion);
      this.outputStride = Number(pnConfigDic.outputStride);
      this.scaleFactor = Number(pnConfigDic.scaleFactor);
      this.minScore = Number(pnConfigDic.minScore);

      this.scaleWidth = getValidResolution(this.scaleFactor, width, this.outputStride);
      this.scaleHeight = getValidResolution(this.scaleFactor, height, this.outputStride);
      this.inputTensor = new Float32Array(this.scaleWidth * this.scaleHeight * channels);
      this.scaleInputSize = [1, this.scaleWidth, this.scaleHeight, channels];

      let HEATMAP_TENSOR_SIZE;
      if ((this.modelVersion == 0.75 || this.modelVersion == 0.5) && this.outputStride == 32) {
        HEATMAP_TENSOR_SIZE = product(toHeatmapsize(this.scaleInputSize, 16));
      } else {
        HEATMAP_TENSOR_SIZE = product(toHeatmapsize(this.scaleInputSize, this.outputStride));
      }
      let OFFSET_TENSOR_SIZE = HEATMAP_TENSOR_SIZE * 2;
      this.heatmapTensor = new Float32Array(HEATMAP_TENSOR_SIZE);
      this.offsetTensor = new Float32Array(OFFSET_TENSOR_SIZE);
      // prepare canvas for predict
      let poseCanvasPredict = document.getElementById('poseCanvasPredict');
      drawContent = await this.loadImage(poseCanvasPredict, width, height);
      width = this.scaleWidth;
      height = this.scaleHeight;
    }

    canvasElement.setAttribute("width", width);
    canvasElement.setAttribute("height", height);
    let canvasContext = canvasElement.getContext('2d');
    canvasContext.drawImage(drawContent, 0, 0, width, height);
    let pixels = canvasContext.getImageData(0, 0, width, height).data;
    if (canvasElement.width !== width || canvasElement.height !== height) {
      throw new Error(`canvasElement.width(${canvasElement.width}) is not ${width} or canvasElement.height(${canvasElement.height}) is not ${height}`);
    }
    if (norm) {
      pixels = new Float32Array(pixels).map(p => p / 255);
    }

    if (channelScheme === 'RGB') {
      // NHWC layout
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let value = pixels[y * width * imageChannels + x * imageChannels + c];
            this.inputTensor[y * width * channels + x * channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else if (channelScheme === 'BGR') {
      // NHWC layout
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let value = pixels[y * width * imageChannels + x * imageChannels + (channels - c - 1)];
            this.inputTensor[y * width * channels + x * channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else {
      throw new Error(`Unknown color channel scheme ${channelScheme}`);
    }
  }

  prepareoutputTensor(outputBoxTensor, outputClassScoresTensor) {
    const outH = [1083, 600, 150, 54, 24, 6];
    const boxLen = 4;
    const classLen = 91;
    let boxOffset = 0;
    let classOffset = 0;
    let boxTensor;
    let classTensor;
    for (let i = 0; i < 6; ++i) {
      boxTensor = outputBoxTensor.subarray(boxOffset, boxOffset + boxLen * outH[i]);
      classTensor = outputClassScoresTensor.subarray(classOffset, classOffset + classLen * outH[i]);
      this.outputTensor[2 * i] = boxTensor;
      this.outputTensor[2 * i + 1] = classTensor;
      boxOffset += boxLen * outH[i];
      classOffset += classLen * outH[i];
    }
  }

  async setupAsync() {
    await this.setInputOutput();
    let backend = this.configuration.backend.replace('native', 'WebML');
    let modelName = this.configuration.modelName;
    if (tfliteModelArray.indexOf(modelName) !== -1) {
      let model = getModelDicItem(modelName);
      let resultTflite = await this.loadModelAndLabels(model);
      this.labels = resultTflite.text.split('\n');
      let flatBuffer = new flatbuffers.ByteBuffer(resultTflite.bytes);
      let rawModel = tflite.Model.getRootAsModel(flatBuffer);
      let postOptions = model.postOptions || {};
      let kwargs = {
        rawModel: rawModel,
        backend: backend,
        prefer: getPreferString(backend),
        softmax: postOptions.softmax || false,
      };
      this.model = new TFliteModelImporter(kwargs);
    } else if (onnxModelArray.indexOf(modelName) !== -1) {
      let model = getModelDicItem(modelName);
      let resultONNX = await this.loadModelAndLabels(model);
      this.labels = resultONNX.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let err = onnx.ModelProto.verify(resultONNX.bytes);
      if (err) {
        throw new Error(`Invalid model ${err}`);
      }
      let rawModel = onnx.ModelProto.decode(resultONNX.bytes);
      let postOptions = model.postOptions || {};
      let kwargs = {
        rawModel: rawModel,
        backend: backend,
        prefer: getPreferString(backend),
        softmax: postOptions.softmax || false,
      };
      this.model = new OnnxModelImporter(kwargs);
    } else if (ssdModelArray.indexOf(modelName) !== -1) {
      let model = getModelDicItem(modelName);
      let resultTflite = await this.loadModelAndLabels(model);
      this.labels = resultTflite.text.split('\n');
      let flatBuffer = new flatbuffers.ByteBuffer(resultTflite.bytes);
      let rawModel = tflite.Model.getRootAsModel(flatBuffer);
      let kwargs = {
        rawModel: rawModel,
        backend: backend,
        prefer: getPreferString(backend),
      };
      this.model = new TFliteModelImporter(kwargs);
    } else if (modelName === 'posenet') {
      let modelArch = ModelArch.get(this.modelVersion);
      let smType = 'Singleperson';
      let cacheMap = new Map();
      let useAtrousConv = false; // Default false, NNAPI and BNNS don't support AtrousConv
      this.model = new PoseNet(modelArch, this.modelVersion, useAtrousConv, this.outputStride,
                               this.scaleInputSize, smType, cacheMap, backend, getPreferString(backend));
    }
    await this.model.createCompiledModel();
  }
  printPredictResult() {
    let probs = Array.from(this.outputTensor);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
      if (a[0] === b[0]) {
        return 0;
      }
      return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    let classes = [];
    for (let i = 0; i < 3; ++i) {
      let prob = sorted[i][0];
      let index = sorted[i][1];
      console.log(`label: ${this.labels[index]}, probability: ${(prob * 100).toFixed(2)}%`);
    }
  }
  async executeSingleAsync() {
    let result;
    result = await this.model.compute([this.inputTensor], [this.outputTensor]);
    console.log(`compute result: ${result}`);
  }
  async executeSingleAsyncSSDMN() {
    let result;
    result = await this.model.compute([this.inputTensor], this.outputTensor);
    console.log(`compute result: ${result}`);
  }
  async executeSingleAsyncPN() {
    let result;
    result = await this.model.computeSinglePose(this.inputTensor, this.heatmapTensor, this.offsetTensor);
    console.log(`compute result: ${result}`);
  }
  async finalizeAsync() {
    this.inputTensor = null;
    this.outputTensor = null;
    this.modelVersion = null;
    this.outputStride = null;
    this.scaleFactor = null;
    this.minScore = null;
    this.scaleWidth = null;
    this.scaleHeight = null;
    this.scaleInputSize = null;
    this.heatmapTensor = null;
    this.offsetTensor  = null;
    this.outputBoxTensor = null;
    this.outputClassScoresTensor = null;
    this.anchors = null;
    this.model = null;
    this.labels = null;
    inputElement.removeAttribute('disabled');
    pickBtnEelement.setAttribute('class', 'btn btn-primary');
  }
}
//---------------------------------------------------------------------------------------------------
//
// Main
//
const BenchmarkClass = {
  'webml-polyfill.js': WebMLJSBenchmark,
  'WebML API': WebMLJSBenchmark
};
async function run() {
  inputElement.setAttribute('class', 'disabled');
  pickBtnEelement.setAttribute('class', 'btn btn-primary disabled');
  let logger = new Logger(document.querySelector('#log'));
  logger.group('Benchmark');
  try {
    let configuration = JSON.parse(document.querySelector('#configurations').selectedOptions[0].value);
    configuration.modelName = document.querySelector('#modelName').selectedOptions[0].value;
    configuration.iteration = Number(document.querySelector('#iteration').value) + 1;
    logger.group('Environment Information');
    logger.log(`${'UserAgent'.padStart(12)}: ${(navigator.userAgent) || '(N/A)'}`);
    logger.log(`${'Platform'.padStart(12)}: ${(navigator.platform || '(N/A)')}`);
    logger.groupEnd();
    logger.group('Configuration');
    Object.keys(configuration).forEach(key => {
      if (key === 'backend' && configuration[key] === 'native') {
        let selectedOpt = preferSelectElement.options[preferSelectElement.selectedIndex];
        logger.log(`${key.padStart(12)}: ${getNativeAPI(selectedOpt.value)}(${selectedOpt.text})`);
      } else if (key === 'modelName') {
        let model = getModelDicItem(configuration[key]);
        logger.log(`${key.padStart(12)}: ${model.modelName}`);
      } else {
        logger.log(`${key.padStart(12)}: ${configuration[key]}`);
      }
    });
    logger.groupEnd();
    logger.group('Run');
    let benchmark = new BenchmarkClass[configuration.framework]();
    benchmark.onExecuteSingle = (i => logger.log(`Iteration: ${i + 1} / ${configuration.iteration}`));
    let summary = await benchmark.runAsync(configuration);
    logger.groupEnd();
    logger.group('Result');
    logger.log(`Inference Time: <em style="color:green;font-weight:bolder;">${summary.computeResults.mean.toFixed(2)}+-${summary.computeResults.std.toFixed(2)}</em> [ms]`);
    if (summary.decodeResults !== null) {
      logger.log(`Decode Time: <em style="color:green;font-weight:bolder;">${summary.decodeResults.mean.toFixed(2)}+-${summary.decodeResults.std.toFixed(2)}</em> [ms]`);
    }
    logger.groupEnd();
  } catch (err) {
    logger.error(err);
  }
  logger.groupEnd();
}
document.addEventListener('DOMContentLoaded', () => {
  inputElement = document.getElementById('input');
  pickBtnEelement = document.getElementById('pickButton');
  imageElement = document.getElementById('image');
  canvasElement = document.getElementById('canvas');
  poseCanvas = document.getElementById('poseCanvas');
  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
      bkPoseImageSrc = imageElement.src;
    }
  }, false);

  let modelElement = document.getElementById('modelName');
  modelElement.addEventListener('change', (e) => {
    bkPoseImageSrc = null;
    let modelName = modelElement.options[modelElement.selectedIndex].value;
    let inputFile = document.getElementById('input').files[0];
    if (inputFile !== undefined) {
      imageElement.src = URL.createObjectURL(inputFile);
    } else {
      if (modelName === 'posenet') {
        imageElement.src = document.getElementById('poseImage').src;
      } else if (ssdModelArray.indexOf(modelName) !== -1) {
        imageElement.src = document.getElementById('ssdMobileImage').src;
      } else {
        imageElement.src = document.getElementById('imageClassificationImage').src;
      }
    }
  }, false);

  let configurationsElement = document.getElementById('configurations');
  configurationsElement.addEventListener('change', (e) => {
    bkPoseImageSrc = null;
    let modelName = modelElement.options[modelElement.selectedIndex].value;
    let inputFile = document.getElementById('input').files[0];
    if (inputFile !== undefined) {
      imageElement.src = URL.createObjectURL(inputFile);
    } else {
      if (modelName === 'posenet') {
        imageElement.src = document.getElementById('poseImage').src;
      } else if (ssdModelArray.indexOf(modelName) !== -1) {
        imageElement.src = document.getElementById('ssdMobileImage').src;
      } else {
        imageElement.src = document.getElementById('imageClassificationImage').src;
      }
    }
    let currentConfig = configurationsElement.options[configurationsElement.selectedIndex].text;
    if (currentConfig.indexOf('WebML API') !== -1) {
      preferDivElement.setAttribute('class', "prefer-show");
      if (currentOS === 'Mac OS') {
        for (var i=0; i<preferSelectElement.options.length; i++) {
          let preferOpt = preferSelectElement.options[i];
          if (preferOpt.value === 'low') {
            preferOpt.disabled = true;
          }
        }
      } else if (['Windows', 'Linux'].indexOf(currentOS) !== -1) {
        for (var i=0; i<preferSelectElement.options.length; i++) {
          let preferOpt = preferSelectElement.options[i];
          if (preferOpt.value === 'sustained') {
            preferOpt.selected = true;
          } else if (preferOpt.value === 'low') {
            preferOpt.disabled = true;
          }
        }
      }
    } else {
      preferDivElement.setAttribute('class', "prefer-hidden");
    }
  }, false);

  let webmljsConfigurations = [{
    framework: 'webml-polyfill.js',
    backend: 'WASM',
    modelName: 'mobilenet_v1_1.0_224.tflite',
    iteration: 0
  },
  {
    framework: 'webml-polyfill.js',
    backend: 'WebGL',
    modelName: 'mobilenet_v1_1.0_224.tflite',
    iteration: 0
  }];
  let webmlAPIConfigurations = [{
    framework: 'WebML API',
    backend: 'native',
    modelName: 'mobilenet_v1_1.0_224.tflite',
    iteration: 0
  }];
  let configurations = [];
  configurations = configurations.concat(webmljsConfigurations, webmlAPIConfigurations);
  for (let configuration of configurations) {
    let option = document.createElement('option');
    option.value = JSON.stringify(configuration);
    option.textContent = configuration.framework + ' (' + configuration.backend + ' backend)';
    if (configuration.framework === 'WebML API') {
      if (navigator.ml.isPolyfill) {
        option.disabled = true;
      }
      if (['Android', 'Windows', 'Linux'].indexOf(currentOS) !== -1) {
        for (var i=0; i<preferSelectElement.options.length; i++) {
          let preferOp = preferSelectElement.options[i];
          if (preferOp.text === 'sustained') {
            preferOp.selected = true;
          } else if (preferOp.text === 'fast') {
            preferOp.disabled = true;
          }
        }
      }
    }
    document.querySelector('#configurations').appendChild(option);
  }
  let button = document.querySelector('#runButton');
  button.setAttribute('class', 'btn btn-primary');
  button.addEventListener('click', run);
});
