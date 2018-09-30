const INPUT_TENSOR_SIZE = 300*300*3;
const MODEL_FILE = './model/ssd_mobilenet.tflite';
const LABELS_FILE = './model/coco_labels_list.txt';

class Utils {
  constructor() {
    this.tfModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputBoxTensor;
    this.outputClassScoresTensor;
    this.anchors;

    this.inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    this.outputBoxTensor = new Float32Array(NUM_BOXES * BOX_SIZE);
    this.outputClassScoresTensor = new Float32Array(NUM_BOXES * NUM_CLASSES);
    this.container = document.getElementById('container');
    this.progressBar = document.getElementById('progressBar');
    this.progressContainer = document.getElementById('progressContainer');
    this.canvasElement = document.getElementById('canvas');
    this.canvasContext = this.canvasElement.getContext('2d');
    this.canvasShowElement = document.getElementById('canvasShow');

    this.initialized = false;
  }

  async init(backend) {
    this.initialized = false;
    let result;
    this.anchors = generateAnchors({});
    if (!this.tfModel) {
      result = await this.loadModelAndLabels(MODEL_FILE, LABELS_FILE);
      this.container.removeChild(progressContainer);
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
      // printTfLiteModel(this.tfModel);
    }
    this.model = new SsdMobileNet(this.tfModel, backend);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute(this.inputTensor, this.outputBoxTensor, this.outputClassScoresTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict(imageSource) {
    if (!this.initialized) return;
    this.canvasContext.drawImage(imageSource, 0, 0,
                                 this.canvasElement.width,
                                 this.canvasElement.height);
    // console.log('inputTensor1', this.inputTensor)
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    // console.log('inputTensor2', this.inputTensor)
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputBoxTensor, this.outputClassScoresTensor);
    // console.log('outputBoxTensor', this.outputBoxTensor)
    // console.log('outputClassScoresTensor', this.outputClassScoresTensor)
    // let startDecode = performance.now();
    decodeOutputBoxTensor(this.outputBoxTensor, this.anchors);
    // console.log(`Decode time: ${(performance.now() - startDecode).toFixed(2)} ms`);
    // let startNMS = performance.now();
    let [totalDetections, boxesList, scoresList, classesList] = NMS({}, this.outputBoxTensor, this.outputClassScoresTensor);
    // console.log(`NMS time: ${(performance.now() - startNMS).toFixed(2)} ms`);
    // let startVisual = performance.now();
    visualize(this.canvasShowElement, totalDetections, imageSource, boxesList, scoresList, classesList, this.labels);
    // console.log(`visual time: ${(performance.now() - startVisual).toFixed(2)} ms`);
    let elapsed = performance.now() - start;
    console.log(`Inference time: ${elapsed.toFixed(2)} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <em style="color:green;font-weight:bloder;">${elapsed.toFixed(2)} </em>ms`;
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
    const width = 300;
    const height = 300;
    const channels = 3;
    const imageChannels = 4; // RGBA
    const mean = 127.5;
    const std = 127.5;
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) or canvas.height(${canvas.height}) is not 300`);
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
}