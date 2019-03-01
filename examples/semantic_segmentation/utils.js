class Utils {
  constructor() {
    this.tfModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;
    this.updateProgress;
    this.modelFile;
    this.labelsFile;
    this.inputSize;
    this.outputSize;
    this.preOptions;
    this.postOptions;
    // this.preprocessCanvas = new OffscreenCanvas(224, 224);
    this.preprocessCanvas = document.createElement('canvas');
    this.preprocessCtx = this.preprocessCanvas.getContext('2d');    

    this.initialized = false;
  }

  async init(backend, prefer) {
    this.initialized = false;
    let result;
    if (!this.tfModel) {
      result = await this.loadModelAndLabels(this.modelFile, this.labelsFile);
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
      printTfLiteModel(this.tfModel);
    }
    let kwargs = {
      rawModel: this.tfModel,
      backend: backend,
      prefer: prefer,
    };
    this.model = new TFliteModelImporter(kwargs);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict() {
    if (!this.initialized) return;
    let start = performance.now();
    await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    return {
      time: elapsed,
      segMap: {
        data: this.outputTensor,
        outputShape: this.outputSize,
        labels: this.labels,
      },
    };
  }

  async loadModelAndLabels(modelUrl, labelsUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(labelsUrl);
    return {bytes: bytes, text: text};
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      if (this.outstandingRequest) {
        this.outstandingRequest.abort();
      }
      let request = new XMLHttpRequest();
      this.outstandingRequest = request;
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        this.outstandingRequest = null;
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

  getFittedResolution(aspectRatio) {
    const height = this.inputSize[0];
    const width = this.inputSize[1];

    // aspectRatio = width / height
    if (aspectRatio > 1) {
      return [width, Math.floor(height / aspectRatio)];
    } else {
      return [Math.floor(width / aspectRatio), height];
    }
  }

  prepareInput(imgSrc) {
    const height = this.inputSize[0];
    const width = this.inputSize[1];

    this.preprocessCanvas.width = width;
    this.preprocessCanvas.height = height;

    let imWidth = imgSrc.naturalWidth | imgSrc.videoWidth;
    let imHeight = imgSrc.naturalHeight | imgSrc.videoHeight;
    // assume deeplab_out.width == deeplab_out.height
    let resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
    let scaledWidth = Math.floor(imWidth / resizeRatio);
    let scaledHeight = Math.floor(imHeight / resizeRatio);

    // better to keep resizeRatio at 1
    // avoid scaling images in `drawImage`, especially in real time scenarios
    this.preprocessCtx.drawImage(imgSrc, 0, 0, scaledWidth, scaledHeight);
    const pixels = this.preprocessCtx.getImageData(0, 0, width, height).data;

    const tensor = this.inputTensor;
    const channels = 3;
    const imageChannels = 4;

    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = value / 127.5 - 1;
        }
      }
    }

    return [scaledWidth, scaledHeight];
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
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    this.numClasses = newModel.numClasses;
    this.inputTensor = new Float32Array(newModel.inputSize.reduce((x,y) => x*y));
    this.outputTensor = new Float32Array(newModel.outputSize.reduce((x,y) => x*y));
    this.tfModel = null;
  }

  // for debugging
  async iterateLayers(configs, layerList) {
    if (!this.initialized) return;

    let iterators = [];
    for (let config of configs) {
      let model = await new TFliteModelImporter({
        rawModel: this.tfModel,
        backend: config.backend,
        prefer: config.prefer || null,
      });
      iterators.push(model.layerIterator([this.inputTensor], layerList));
    }

    while (true) {

      let layerOutputs = [];
      for (let it of iterators) {
        layerOutputs.push(await it.next());
      }

      let refOutput = layerOutputs[0];
      if (refOutput.done) {
        break;
      }

      console.debug(`\n\n\nLayer(${refOutput.value.layerId}) ${refOutput.value.outputName}`);

      for (let i = 0; i < configs.length; ++i) {
        console.debug(`\n${configs[i].backend}:`);
        console.debug(`\n${layerOutputs[i].value.tensor}`);

        if (i > 0) {
          let sum = 0;
          for (let j = 0; j < refOutput.value.tensor.length; j++) {
            sum += Math.pow(layerOutputs[i].value.tensor[j] - refOutput.value.tensor[j], 2);
          }
          let variance = sum / refOutput.value.tensor.length;
          console.debug(`var with ${configs[0].backend}: ${variance}`);
        }
      }
    }
  }
}
