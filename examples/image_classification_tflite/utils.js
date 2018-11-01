class Utils {
  constructor(canvas) {
    this.tfModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;
    this.MODEL_FILE;
    this.LABELS_FILE;
    this.INPUT_SIZE;
    this.OUTPUT_SIZE;
    this.canvasElement = canvas;
    this.canvasContext = this.canvasElement.getContext('2d');
    this.updateProgress;
    this.initialized = false;
  }

  async init(backend) {
    this.initialized = false;
    let result;
    if (!this.tfModel) {
      result = await this.loadModelAndLabels(this.MODEL_FILE, this.LABELS_FILE);
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
      printTfLiteModel(this.tfModel);
    }
    this.model = new ImageClassificationModel(this.tfModel, backend);
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
    return {
      time: elapsed.toFixed(2),
      classes: this.getTopClasses(this.outputTensor, this.labels, 3)
    }
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
          if (self.updateProgress) {
            self.updateProgress(ev);
        }
        };
      }
      request.send();
    });
  }

  prepareInputTensor(tensor, canvas) {
    const width = this.INPUT_SIZE[1];
    const height = this.INPUT_SIZE[0];
    const channels = 3;
    const imageChannels = 4; // RGBA
    const mean = 127.5;
    const std = 127.5;
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) is not ${this.INPUT_SIZE[1]} or canvas.height(${canvas.height}) is not ${this.INPUT_SIZE[0]}`);
    }
    let context = canvas.getContext('2d');
    let pixels = context.getImageData(0, 0, width, height).data;
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = (value - mean)/std;
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

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

  changeModelParam (newModel) {
    this.INPUT_SIZE = newModel.INPUT_SIZE;
    this.OUTPUT_SIZE = newModel.OUTPUT_SIZE;
    this.MODEL_FILE = newModel.MODEL_FILE;
    this.LABELS_FILE = newModel.LABELS_FILE;
    this.inputTensor = new Float32Array(this.INPUT_SIZE.reduce((a, b) => a * b));
    this.outputTensor = new Float32Array(this.OUTPUT_SIZE);
    this.tfModel = null;

    this.canvasElement.width = newModel.INPUT_SIZE[1];
    this.canvasElement.height = newModel.INPUT_SIZE[0];
  }
}
