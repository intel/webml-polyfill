class Utils {
  constructor(canvas) {
    this.rawModel;
    this.model;
    this.inputTensor = [];
    this.outputTensor = [];
    this.inputSize;
    this.outputSize;
    this.preOptions;
    this.postOptions;
    this.canvasElement = canvas;
    this.canvasContext = this.canvasElement.getContext('2d');
    this.updateProgress;
    this.backend = '';
    this.prefer = '';
    this.initialized = false;
    this.loaded = false;
    this.resolveGetRequiredOps = null;
    this.outstandingRequest = null;
  }

  async loadModel(newModel) {
    if (this.loaded && this.modelFile === newModel.modelFile) {
      return 'LOADED';
    }
    // reset all states
    this.loaded = this.initialized = false;
    this.backend = this.prefer = '';

    // set new model params
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.modelFile = newModel.modelFile;
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    this.numClasses = newModel.numClasses;
    this.inputTensor = [new Float32Array(newModel.inputSize.reduce((x,y) => x*y))];
    this.outputTensor = [new Float32Array(newModel.outputSize)];

    let result = await this.loadRawModel(this.modelFile);
    let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
    this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
    printTfLiteModel(this.tfModel);

    this.loaded = true;
    return 'SUCCESS';
  }


  async init(backend, prefer) {
    if (!this.loaded) {
      return 'NOT_LOADED';
    }
    if (this.initialized && backend === this.backend && prefer === this.prefer) {
      return 'INITIALIZED';
    }
    this.initialized = false;
    this.backend = backend;
    this.prefer = prefer;
    let kwargs = {
      rawModel: this.tfModel,
      backend: backend,
      prefer: prefer,
    };
    this.model = new TFliteModelImporter(kwargs);
    let result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;

    if (this.resolveGetRequiredOps) {
      this.resolveGetRequiredOps(this.model.getRequiredOps());
    }

    return 'SUCCESS';
  }

  async getRequiredOps() {
    if (!this.initialized) {
      return new Promise(resolve => this.resolveGetRequiredOps = resolve);
    } else {
      return this.model.getRequiredOps();
    }
  }

  getSubgraphsSummary() {
    if (this.model._backend !== 'WebML' &&
        this.model &&
        this.model._compilation &&
        this.model._compilation._preparedModel) {
      return this.model._compilation._preparedModel.getSubgraphsSummary();
    } else {
      return [];
    }
  }

  async predict(imageSource, box) {
    if (!this.initialized) return;
    this.canvasContext.drawImage(imageSource, box[0], box[2], 
                                 box[1]-box[0], box[3]-box[2], 0, 0, 
                                 this.canvasElement.width,
                                 this.canvasElement.height);
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`Landmark Detection Inference time: ${elapsed.toFixed(2)} ms`);
    let outputTensor = this.outputTensor[0];
    return {keyPoints: outputTensor, time: elapsed.toFixed(2)};
  }

  async loadRawModel(modelUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    return {bytes: bytes};
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

  prepareInputTensor(tensors, canvas) {
    let tensor = tensors[0];
    const width = this.inputSize[1];
    const height = this.inputSize[0];
    const channels = this.inputSize[2];
    const imageChannels = 4; // RGBA
    const mean = this.preOptions.mean || [0, 0, 0, 0];
    const std  = this.preOptions.std  || [1, 1, 1, 1];
    const norm = this.preOptions.norm || false;

    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) is not ${width} or canvas.height(${canvas.height}) is not ${height}`);
    }
    let context = canvas.getContext('2d');
    let pixels = context.getImageData(0, 0, width, height).data;
    // NHWC layout
    if (norm) {
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let value = pixels[y*width*imageChannels + x*imageChannels + c] / 255;
            tensor[y*width*channels + x*channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else {
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let value = pixels[y*width*imageChannels + x*imageChannels + c];
            tensor[y*width*channels + x*channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    }
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }
}