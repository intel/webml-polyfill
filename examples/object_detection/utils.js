class Utils {
  constructor(canvas, canvasShow) {
    this.rawModel;
    this.labels;
    this.model;
    this.inputTensor = [];
    this.outputTensor = [];
    this.outputBoxTensor;
    this.outputClassScoresTensor;
    this.inputSize;
    this.outputSize;
    this.preOptions;
    this.postOptions;
    this.boxSize;
    this.numClasses;
    this.numBoxes;
    this.anchors;
    this.canvasElement = canvas;
    this.canvasContext = this.canvasElement.getContext('2d');
    this.canvasShowElement = canvasShow;
    this.updateProgress;

    this.initialized = false;
  }

  async init(backend, prefer) {
    this.initialized = false;
    let result;
    this.anchors = generateAnchors({});
    if (!this.rawModel) {
      result = await this.loadModelAndLabels(this.modelFile, this.labelsFile);
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.rawModel = tflite.Model.getRootAsModel(flatBuffer);
      // printTfLiteModel(this.rawModel);
    }
    let kwargs = {
      rawModel: this.rawModel,
      backend: backend,
      prefer: prefer,
    };
    this.model = new TFliteModelImporter(kwargs);
    this.prepareoutputTensor(this.outputBoxTensor, this.outputClassScoresTensor);
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
    // console.log('inputTensor1', this.inputTensor)
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    // console.log('inputTensor2', this.inputTensor)
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    // console.log('outputBoxTensor', this.outputBoxTensor)
    // console.log('outputClassScoresTensor', this.outputClassScoresTensor)
    // let startDecode = performance.now();
    decodeOutputBoxTensor({}, this.outputBoxTensor, this.anchors);
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
              reject(new Error('Failed to load ' + url + ' status: ' + request.status));
          }
        }
      };
      if (progress && typeof this.updateProgress !== 'undefined') {
        request.onprogress = this.updateProgress;
      }
      request.send();
    });
  }

  prepareInputTensor(tensors, canvas) {
    let tensor = tensors[0];
    const width = this.inputSize[1];
    const height = this.inputSize[0];
    const channels = this.inputSize[2];
    const imageChannels = 4; // RGBA
    const mean = this.preOptions.mean || [0, 0, 0, 0];
    const std  = this.preOptions.std  || [1, 1, 1, 1];

    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) is not ${width} or canvas.height(${canvas.height}) is not ${height}`);
    }
    let context = canvas.getContext('2d');
    let pixels = context.getImageData(0, 0, width, height).data;
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = (value - mean[c])/std[c];
        }
      }
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

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

  changeModelParam(newModel) {
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.modelFile = newModel.modelFile;
    this.labelsFile = newModel.labelsFile;
    this.boxSize = newModel.box_size;
    this.numClasses = newModel.num_classes;
    this.numBoxes = newModel.num_boxes;
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    this.inputTensor[0] = new Float32Array(this.inputSize.reduce((a, b) => a * b));
    this.outputBoxTensor = new Float32Array(this.numBoxes * this.boxSize);
    this.outputClassScoresTensor = new Float32Array(this.numBoxes * this.numClasses);
    this.rawModel = null;

    this.canvasElement.width = newModel.inputSize[1];
    this.canvasElement.height = newModel.inputSize[0];
  }
}