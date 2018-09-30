'use strict';
const MODEL_DIC = {
  mobilenet: {
    width: 224,
    height: 224,
    inputTensorSize: 224 * 224 * 3,
    outputTensorSize: 1001,
    modelFile: '../examples/mobilenet/model/mobilenet_v1_1.0_224.tflite',
    labelFile: '../examples/mobilenet/model/labels.txt'
  },
  squeezenet: {
    width: 224,
    height: 224,
    inputTensorSize: 224 * 224 * 3,
    outputTensorSize: 1000,
    modelFile: '../examples/squeezenet/model/model.onnx',
    labelFile: '../examples/squeezenet/labels.json'
  },
  posenet: {
    width: 513,
    height: 513
  },
  ssdmobilenet: {
    width: 300,
    height: 300,
    inputTensorSize: 300 * 300 * 3,
    outputBoxTensorSize: (1083 + 600 + 150 + 54 + 24 + 6) * 4,
    outputClassScoresTensorSize: (1083 + 600 + 150 + 54 + 24 + 6) * 91,
    modelFile: '../examples/ssd_mobilenet/model/ssd_mobilenet.tflite',
    labelFile: '../examples/ssd_mobilenet/model/coco_labels_list.txt'
  },
}
let imageElement = null;
let inputElement = null;
let pickBtnEelement = null;
let canvasElement = null;
let poseCanvas = null;
let bkPoseImageSrc = null;
let pnConfigDic = null;
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
    if (this.configuration.modelName === 'mobilenet' || this.configuration.modelName === 'squeezenet') {
      for (let i = 0; i < this.configuration.iteration; i++) {
        this.onExecuteSingle(i);
        await new Promise(resolve => requestAnimationFrame(resolve));
        let tStart = performance.now();
        await this.executeSingleAsync();
        let elapsedTime = performance.now() - tStart;
        this.printPredictResult();
        computeResults.push(elapsedTime);
      }
    } else if (this.configuration.modelName === 'posenet') {
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
    } else if (this.configuration.modelName === 'ssdmobilenet') {
      for (let i = 0; i < this.configuration.iteration; i++) {
        this.onExecuteSingle(i);
        await new Promise(resolve => requestAnimationFrame(resolve));
        let tStart = performance.now();
        await this.executeSingleAsyncSSDMN();
        let elapsedTime = performance.now() - tStart;
        computeResults.push(elapsedTime);
        let dstart = performance.now();
        decodeOutputBoxTensor(this.outputBoxTensor, this.anchors);
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
  async loadModelAndLabels() {
    let arrayBuffer = await this.loadUrl(MODEL_DIC[this.configuration.modelName].modelFile, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(MODEL_DIC[this.configuration.modelName].labelFile);
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
    const channels = 3;
    const imageChannels = 4; // RGBA
    let configModelName = this.configuration.modelName;
    let width = MODEL_DIC[configModelName].width;;
    let height = MODEL_DIC[configModelName].height;
    let drawContent;
    if (configModelName === 'mobilenet' || configModelName === 'squeezenet') {
      this.inputTensor = new Float32Array(MODEL_DIC[configModelName].inputTensorSize);
      this.outputTensor = new Float32Array(MODEL_DIC[configModelName].outputTensorSize);
      drawContent = imageElement;
    } else if (configModelName === 'ssdmobilenet') {
      if (bkPoseImageSrc !== null) {
        // reset for rerun with same image
        imageElement.src = bkPoseImageSrc;
      }
      this.inputTensor = new Float32Array(MODEL_DIC[configModelName].inputTensorSize);
      this.outputBoxTensor = new Float32Array(MODEL_DIC[configModelName].outputBoxTensorSize);
      this.outputClassScoresTensor = new Float32Array(MODEL_DIC[configModelName].outputClassScoresTensorSize);
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
      this.inputTensor = new Float32Array(this.scaleWidth * this.scaleHeight * 3);
      this.scaleInputSize = [1, this.scaleWidth, this.scaleHeight, 3];

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
    if (configModelName === 'mobilenet' || configModelName === 'posenet' || configModelName === 'ssdmobilenet') {
      const meanMN = 127.5;
      const stdMN = 127.5;
      // NHWC layout
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let value = pixels[y * width * imageChannels + x * imageChannels + c];
            this.inputTensor[y * width * channels + x * channels + c] = (value - meanMN) / stdMN;
          }
        }
      }
    } else if (configModelName === 'squeezenet') {
      // The RGB mean values are from
      // https://github.com/caffe2/AICamera/blob/master/app/src/main/cpp/native-lib.cpp#L108
      const meanSN = [122.67891434, 116.66876762, 104.00698793];
      // NHWC layout
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let value = pixels[y * width * imageChannels + x * imageChannels + c];
            this.inputTensor[y * width * channels + x * channels + c] = value - meanSN[c];
          }
        }
      }
    }
  }
  async setupAsync() {
    await this.setInputOutput();
    let targetModel;
    if (this.configuration.modelName === 'mobilenet') {
      let resultMN = await this.loadModelAndLabels();
      this.labels = resultMN.text.split('\n');
      let flatBuffer = new flatbuffers.ByteBuffer(resultMN.bytes);
      targetModel = tflite.Model.getRootAsModel(flatBuffer);
      if (this.configuration.backend !== 'native') {
        this.model = new MobileNet(targetModel, this.configuration.backend);
      } else {
        this.model = new MobileNet(targetModel);
      }
    } else if (this.configuration.modelName === 'ssdmobilenet') {
      let resultSSDMN = await this.loadModelAndLabels();
      this.labels = resultSSDMN.text.split('\n');
      let flatBuffer = new flatbuffers.ByteBuffer(resultSSDMN.bytes);
      targetModel = tflite.Model.getRootAsModel(flatBuffer);
      if (this.configuration.backend !== 'native') {
        this.model = new SsdMobileNet(targetModel, this.configuration.backend);
      } else {
        this.model = new SsdMobileNet(targetModel);
      }
    } else if (this.configuration.modelName === 'squeezenet') {
      let resultSN = await this.loadModelAndLabels();
      this.labels = JSON.parse(resultSN.text);
      let err = onnx.ModelProto.verify(resultSN.bytes);
      if (err) {
        throw new Error(`Invalid model ${err}`);
      }
      targetModel = onnx.ModelProto.decode(resultSN.bytes);
      if (this.configuration.backend !== 'native') {
        this.model = new SqueezeNet(targetModel, this.configuration.backend);
      } else {
        this.model = new SqueezeNet(targetModel);
      }
    } else if (this.configuration.modelName === 'posenet') {
      let modelArch = ModelArch.get(this.modelVersion);
      let smType = 'Singleperson';
      let cacheMap = new Map();
      if (this.configuration.backend !== 'native') {
        this.model = new PoseNet(modelArch, this.modelVersion, this.outputStride,
                                 this.scaleInputSize, smType, cacheMap, this.configuration.backend);
      } else {
        this.model = new PoseNet(modelArch, this.modelVersion, this.outputStride,
                                 this.scaleInputSize, smType, cacheMap);
      }
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
    result = await this.model.compute(this.inputTensor, this.outputTensor);
    console.log(`compute result: ${result}`);
  }
  async executeSingleAsyncSSDMN() {
    let result;
    result = await this.model.compute(this.inputTensor, this.outputBoxTensor, this.outputClassScoresTensor);
    console.log(`compute result: ${result}`);
  }
  async executeSingleAsyncPN() {
    let result;
    result = await this.model.computeSinglePose(this.inputTensor, this.heatmapTensor, this.offsetTensor);
    console.log(`compute result: ${result}`);
  }
  async finalizeAsync() {
    this.model = null;
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
        logger.log(`${key.padStart(12)}: ${getNativeAPI()}`);
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
    logger.log(`Compute Time: <em style="color:green;font-weight:bolder;">${summary.computeResults.mean.toFixed(2)}+-${summary.computeResults.std.toFixed(2)}</em> [ms]`);
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
    let modelName = modelElement.options[modelElement.selectedIndex].text;
    let inputFile = document.getElementById('input').files[0];
    if (inputFile !== undefined) {
      imageElement.src = URL.createObjectURL(inputFile);
    } else {
      if (modelName === 'PoseNet') {
        imageElement.src = document.getElementById('poseImage').src;
      } else if (modelName === 'SSD MobileNet') {
        imageElement.src = document.getElementById('ssdMobileImage').src;
      } else {
        imageElement.src = document.getElementById('mobileImage').src;
      }
    }
  }, false);

  let configurationsElement = document.getElementById('configurations');
  configurationsElement.addEventListener('change', (e) => {
    bkPoseImageSrc = null;
    let modelName = modelElement.options[modelElement.selectedIndex].text;
    let inputFile = document.getElementById('input').files[0];
    if (inputFile !== undefined) {
      imageElement.src = URL.createObjectURL(inputFile);
    } else {
      if (modelName === 'PoseNet') {
        imageElement.src = document.getElementById('poseImage').src;
      } else if (modelName === 'SSD MobileNet') {
        imageElement.src = document.getElementById('ssdMobileImage').src;
      } else {
        imageElement.src = document.getElementById('mobileImage').src;
      }
    }
  }, false);

  let webmljsConfigurations = [{
    framework: 'webml-polyfill.js',
    backend: 'WASM',
    modelName: 'mobilenet',
    iteration: 0
  },
  {
    framework: 'webml-polyfill.js',
    backend: 'WebGL2',
    modelName: 'mobilenet',
    iteration: 0
  }];
  let webmlAPIConfigurations = [{
    framework: 'WebML API',
    backend: 'native',
    modelName: 'mobilenet',
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
    }
    document.querySelector('#configurations').appendChild(option);
  }
  let button = document.querySelector('#runButton');
  button.setAttribute('class', 'btn btn-primary');
  button.addEventListener('click', run);
});
