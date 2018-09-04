'use strict';
const MODEL_DIC = {
  mobilenet: {
    inputTensorSize: 224 * 224 * 3,
    outputTensorSize: 1001,
    modelFile: '../examples/mobilenet/model/mobilenet_v1_1.0_224.tflite',
    labelFile: '../examples/mobilenet/model/labels.txt'
  },
  squeezenet: {
    inputTensorSize: 224 * 224 * 3,
    outputTensorSize: 1000,
    modelFile: '../examples/squeezenet/model/model.onnx',
    labelFile: '../examples/squeezenet/labels.json'
  }
}
let imageElement = null;
let inputElement = null;
let pickBtnEelement = null;
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
    return this.summarize(results);
  }
  /**
   * Setup model
   * @returns {Promise<void>}
   */
  async setupAsync() {
    throw Error("Not Implemented");
  }
  async executeAsync() {
    let results = [];
    for (let i = 0; i < this.configuration.iteration; i++) {
      this.onExecuteSingle(i);
      await new Promise(resolve => requestAnimationFrame(resolve));
      let tStart = performance.now();
      await this.executeSingleAsync();
      let elapsedTime = performance.now() - tStart;
      results.push(elapsedTime);
    }
    return results;
  }
  /**
   * Execute model
   * @returns {Promise<void>}
   */
  async executeSingleAsync() {
    throw Error('Not Implemented');
  }
  /**
   * Finalize
   * @returns {Promise<void>}
   */
  async finalizeAsync() {}
  summarize(results) {
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
  }
  onExecuteSingle(iteration) {}
}
class WebMLJSBenchmark extends Benchmark {
  constructor() {
    super(...arguments);
    this.inputTensor = null;
    this.outputTensor = null;
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
  setInputOutput() {
    const width = 224;
    const height = 224;
    const channels = 3;
    const imageChannels = 4; // RGBA
    this.inputTensor = new Float32Array(MODEL_DIC[this.configuration.modelName].inputTensorSize);
    this.outputTensor = new Float32Array(MODEL_DIC[this.configuration.modelName].outputTensorSize);
    let canvasElement = document.getElementById('canvas');
    let canvasContext = canvasElement.getContext('2d');
    canvasContext.drawImage(imageElement, 0, 0, width, height);
    let pixels = canvasContext.getImageData(0, 0, width, height).data;
    if (this.configuration.modelName === 'mobilenet') {
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
    } else if (this.configuration.modelName === 'squeezenet') {
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
    this.setInputOutput();
    let result = await this.loadModelAndLabels();
    let targetModel;
    if (this.configuration.modelName === 'mobilenet') {
      this.labels = result.text.split('\n');
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      targetModel = tflite.Model.getRootAsModel(flatBuffer);
      if (this.configuration.backend !== 'native') {
        this.model = new MobileNet(targetModel, this.configuration.backend);
      } else {
        this.model = new MobileNet(targetModel);
      }
    } else if (this.configuration.modelName === 'squeezenet') {
      this.labels = JSON.parse(result.text);
      let err = onnx.ModelProto.verify(result.bytes);
      if (err) {
        throw new Error(`Invalid model ${err}`);
      }
      targetModel = onnx.ModelProto.decode(result.bytes);
      if (this.configuration.backend !== 'native') {
        this.model = new SqueezeNet(targetModel, this.configuration.backend);
      } else {
        this.model = new SqueezeNet(targetModel);
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
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    console.log(`compute result: ${result}`);
    this.printPredictResult();
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
    logger.log(`Elapsed Time: <em style="color:green;font-weight:bolder;">${summary.mean.toFixed(2)}+-${summary.std.toFixed(2)}</em> [ms]`);
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
  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
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
  let os = getOS();
  if (os === 'Android' || os === 'Mac OS' ) {
    configurations = configurations.concat(webmljsConfigurations, webmlAPIConfigurations);
  } else if (os === 'Windows') {
    configurations = configurations.concat(webmljsConfigurations);
  } else if (os === 'Linux') {
    configurations = configurations.concat(webmljsConfigurations);
  }
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
