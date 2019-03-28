class FaceDetecor {
  constructor(canvas) {
    this.rawModel;
    this.model;
    this.modelType;
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
    this.margin;
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
    this.modelType = newModel.type;
    this.numClasses = newModel.num_classes;
    this.margin = newModel.margin;
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    this.inputTensor = [new Float32Array(this.inputSize.reduce((a, b) => a * b))];
    this.outputBoxTensor = new Float32Array(this.numBoxes * this.boxSize);	    
    if (this.modelType === 'SSD') {
      this.outputClassScoresTensor = new Float32Array(this.numBoxes * this.numClasses);
      this.anchors = generateAnchors({});
      this.boxSize = newModel.box_size;
      this.numBoxes = newModel.num_boxes;
      this.outputBoxTensor = new Float32Array(this.numBoxes * this.boxSize);
      this.outputClassScoresTensor = new Float32Array(this.numBoxes * this.numClasses);
      this.outputTensor = this.prepareSsdOutputTensor(this.outputBoxTensor, this.outputClassScoresTensor);
    } else {
      this.anchors = newModel.anchors;
      this.outputTensor = [new Float32Array(this.outputSize)];
    }
    this.rawModel = null;

    this.canvasElement.width = newModel.inputSize[1];
    this.canvasElement.height = newModel.inputSize[0];

    let result = await this.loadRawModel(this.modelFile);
    let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
    this.rawModel = tflite.Model.getRootAsModel(flatBuffer);
    printTfLiteModel(this.rawModel);

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
    this.backend = backend;
    this.prefer = prefer;
    this.initialized = false;
    let kwargs = {
      rawModel: this.rawModel,
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

  async getFaceBoxes(imageSource) {
    if (this.modelType === 'SSD') {
      let time = await this.predictSSD(imageSource);
      decodeOutputBoxTensor({}, this.outputBoxTensor, this.anchors);
      let [totalDetections, boxesList, scoresList, classesList] = NMS({num_classes: 2}, this.outputBoxTensor, this.outputClassScoresTensor);
      boxesList = cropSSDBox(imageSource, totalDetections, boxesList, this.margin);
      let outputBoxes = [];
      for (let i = 0; i < totalDetections; ++i) {
        let [ymin, xmin, ymax, xmax] = boxesList[i];
        ymin = Math.max(0, ymin) * imageSource.height;
        xmin = Math.max(0, xmin) * imageSource.width;
        ymax = Math.min(1, ymax) * imageSource.height;
        xmax = Math.min(1, xmax) * imageSource.width;
        let prob = 1 / (1 + Math.exp(-scoresList[i]));
        outputBoxes.push([xmin, xmax, ymin, ymax, prob]);
      }
      return {boxes: outputBoxes, time: time};
    }
    else {
      let time = await this.predictYolo(imageSource);
      let decode_out = decodeYOLOv2({nb_class: 1}, this.outputTensor[0], this.anchors);
      let outputBoxes = getBoxes(decode_out, this.margin);
      for (let i = 0; i < outputBoxes.length; ++i) {
        let [xmin, xmax, ymin, ymax, prob] = outputBoxes[i].slice(1, 6);
        xmin = Math.max(0, xmin) * imageSource.width;
        xmax = Math.min(1, xmax) * imageSource.width;
        ymin = Math.max(0, ymin) * imageSource.height;
        ymax = Math.min(1, ymax) * imageSource.height;
        outputBoxes[i] = [xmin, xmax, ymin, ymax, prob];
      }
      return {boxes: outputBoxes, time: time};
    }
  }

  async predictSSD(imageSource) {
    if (!this.initialized) return;
    this.canvasContext.drawImage(imageSource, 0, 0,
                                  this.canvasElement.width,
                                  this.canvasElement.height);
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`Face Detection Inference time: ${elapsed.toFixed(2)} ms`);
    return elapsed.toFixed(2);
  }

  async predictYolo(imageSource) {
    if (!this.initialized) return;
    this.canvasContext.drawImage(imageSource, 0, 0,
                                this.canvasElement.width,
                                this.canvasElement.height);
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`Face Detection Inference time: ${elapsed.toFixed(2)} ms`);
    return elapsed.toFixed(2);
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

  prepareSsdOutputTensor(outputBoxTensor, outputClassScoresTensor) {
    let outputTensor = [];
    const outH = [1083, 600, 150, 54, 24, 6];
    const boxLen = 4;
    const classLen = 2;
    let boxOffset = 0;
    let classOffset = 0;
    let boxTensor;
    let classTensor;
    for (let i = 0; i < 6; ++i) {
      boxTensor = outputBoxTensor.subarray(boxOffset, boxOffset + boxLen * outH[i]);
      classTensor = outputClassScoresTensor.subarray(classOffset, classOffset + classLen * outH[i]);
      outputTensor[2 * i] = boxTensor;
      outputTensor[2 * i + 1] = classTensor;
      boxOffset += boxLen * outH[i];
      classOffset += classLen * outH[i];
    }
    return outputTensor;
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }
}