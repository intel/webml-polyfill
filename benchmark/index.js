'use strict';
const MOBILENET_INPUT_TENSOR_SIZE = 224 * 224 * 3;
const MOBILENET_OUTPUT_TENSOR_SIZE = 1001;
const MOBILENET_MODEL_FILE = '../examples/mobilenet/model/mobilenet_v1_1.0_224.tflite';
const MOBILENET_LABELS_FILE = '../examples/mobilenet/model/labels.txt';
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
    this.$dom.textContent += `\n${'\t'.repeat(this.indent) + message}`;
  }
  error(err) {
    console.error(err);
    this.$dom.textContent += `\n${'\t'.repeat(this.indent) + err.message}`;
  }
  group(name) {
    console.group(name);
    this.log('');
    this.$dom.textContent += `\n${'\t'.repeat(this.indent) + name}`;
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
    let arrayBuffer = await this.loadUrl(MOBILENET_MODEL_FILE, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(MOBILENET_LABELS_FILE);
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
    const mean = 127.5;
    const std = 127.5;
    this.inputTensor = new Float32Array(MOBILENET_INPUT_TENSOR_SIZE);
    this.outputTensor = new Float32Array(MOBILENET_OUTPUT_TENSOR_SIZE);
    //imageElement = document.getElementById('image');
    let canvasElement = document.getElementById('canvas');
    let canvasContext = canvasElement.getContext('2d');
    canvasContext.drawImage(imageElement, 0, 0, width, height);
    let pixels = canvasContext.getImageData(0, 0, width, height).data;
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          //let value = Math.floor(Math.random() * 255) + 1;
          let value = pixels[y * width * imageChannels + x * imageChannels + c];
          this.inputTensor[y * width * channels + x * channels + c] = (value - mean) / std;
        }
      }
    }
  }
  async setupAsync() {
    this.setInputOutput();
    let result = await this.loadModelAndLabels();
    this.labels = result.text.split('\n');
    //console.log(`labels: ${this.labels}`);
    let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
    let tfModel = tflite.Model.getRootAsModel(flatBuffer);
    //printTfLiteModel(tfModel);
    this.model = new MobileNet(tfModel);
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
    this.model.compute(this.inputTensor, this.outputTensor).then(result => {
      console.log(`compute result: ${result}`);
      this.printPredictResult();
    }).catch(e => {
      console.log(e);
    });
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
  'webml-polyfill.js': WebMLJSBenchmark
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
      logger.log(`${key.padStart(12)}: ${configuration[key]}`);
    });
    logger.groupEnd();
    logger.group('Run');
    let benchmark = new BenchmarkClass[configuration.framework]();
    benchmark.onExecuteSingle = (i => logger.log(`Iteration: ${i + 1} / ${configuration.iteration}`));
    let summary = await benchmark.runAsync(configuration);
    logger.groupEnd();
    logger.group('Result');
    logger.log(`Elapsed Time: ${summary.mean.toFixed(2)}+-${summary.std.toFixed(2)}[ms/batch]`);
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
    name: 'webml-polyfill.js (WebAssembly backend)',
    modelName: 'mobilenet',
    iteration: 0
  }];
  let configurations = [];
  configurations = configurations.concat(webmljsConfigurations);
  for (let configuration of configurations) {
    let option = document.createElement('option');
    option.value = JSON.stringify(configuration);
    option.textContent = configuration.name;
    document.querySelector('#configurations').appendChild(option);
  }
  let button = document.querySelector('#runButton');
  button.setAttribute('class', 'btn btn-primary');
  button.addEventListener('click', run);
});
