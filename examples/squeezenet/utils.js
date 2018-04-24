const INPUT_TENSOR_SIZE = 224*224*3;
const OUTPUT_TENSOR_SIZE = 1000;
const MODEL_FILE = './model/model.onnx';
const LABELS_FILE = 'labels.json';

class Utils {
  constructor() {
    this.onnxModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;

    this.inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    this.outputTensor = new Float32Array(OUTPUT_TENSOR_SIZE);
    this.container = document.getElementById('container');
    this.progressBar = document.getElementById('progressBar');
    this.progressContainer = document.getElementById('progressContainer');
    this.canvasElement = document.getElementById('canvas');
    this.canvasContext = this.canvasElement.getContext('2d');

    this.initialized = false;
  }

  async init(backend) {
    this.initialized = false;
    let result;
    if (!this.onnxModel) {
      result = await this.loadModelAndLabels(MODEL_FILE, LABELS_FILE);
      this.container.removeChild(progressContainer);
      this.labels = JSON.parse(result.text);
      console.log(`labels: ${this.labels}`);
      let err = onnx.ModelProto.verify(result.bytes);
      if (err) {
        throw new Error(`Invalid model ${err}`);
      }
      this.onnxModel = onnx.ModelProto.decode(result.bytes);
      printOnnxModel(this.onnxModel);
    }
    this.model = new SqueezeNet(this.onnxModel, backend);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict(imageSource) {
    if (!this.initialized) return;
    this.canvasContext.drawImage(imageSource, 0, 0,
                                 this.canvasElement.width,
                                 this.canvasElement.height);
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    let classes = this.getTopClasses(this.outputTensor, this.labels, 3);
    console.log(`Inference time: ${elapsed.toFixed(2)} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: ${elapsed.toFixed(2)} ms`;
    console.log(`Classes: `);
    classes.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  }

  async loadModelAndLabels(modelUrl, labelsUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(labelsUrl);
    return {bytes: bytes, text: text};
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      let request = new XMLHttpRequest();
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        if (request.readyState === 4) {
          if (request.status === 200) {
              resolve(request.response);
          } else {
              reject(new Error('Failed to load ' + modelUrl + ' status: ' + request.status));
          }
        }
      };
      if (progress) {
        let self = this;
        request.onprogress = function(ev) {
          if (ev.lengthComputable) {
            let percentComplete = ev.loaded / ev.total * 100;
            percentComplete = percentComplete.toFixed(0);
            self.progressBar.style = `width: ${percentComplete}%`;
            self.progressBar.innerHTML = `${percentComplete}%`;
          }
        }
      }
      request.send();
    });
  }

  prepareInputTensor(tensor, canvas) {
    const width = 224;
    const height = 224;
    const channels = 3;
    const imageChannels = 4; // RGBA
    // The RGB mean values are from
    // https://github.com/caffe2/AICamera/blob/master/app/src/main/cpp/native-lib.cpp#L108
    const mean = [122.67891434, 116.66876762, 104.00698793];
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) or canvas.height(${canvas.height}) is not 224`);
    }
    let context = canvas.getContext('2d');
    let pixels = context.getImageData(0, 0, width, height).data;
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = value - mean[c];
        }
      }
    }
  }

  getTopClasses(tensor, labels, k = 5) {
    let probs = Array.from(tensor);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
      if (a[0] === b[0]) {return 0;}
      return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    let classes = [];
    for (let i = 0; i < k; ++i) {
      let prob = sorted[i][0];
      let index = sorted[i][1];
      let c = {
        label: labels[index],
        prob: (prob * 100).toFixed(2)
      }
      classes.push(c);
    }
    return classes;
  }
}
