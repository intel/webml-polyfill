class Utils {
  constructor(canvas, canvasShow) {
    this.rawModel;
    this.labels;
    this.model;
    this.modelType;
    this.outputTensors = [];
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
    this.imageSource;
    this.canvasShowElement = canvasShow;
    this.updateProgress;
    this.loaded = false;
    this.resolveGetRequiredOps = null;
    this.initialized = false;
  }

  async loadModel(newModel) {
    if (this.loaded && this.modelFile === newModel.modelFile) {
      return 'LOADED';
    }
    // reset all states
    this.loaded = this.initialized = false;
    this.backend = this.prefer = '';
    this.imageSource = null;

    // set new model params
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.modelFile = newModel.modelFile;
    this.modelType = newModel.type;
    this.labelsFile = newModel.labelsFile;
    this.numClasses = newModel.num_classes;
    this.margin = newModel.margin;
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    if (this.modelType === 'SSD') {
      this.anchors = generateAnchors({});
      this.boxSize = newModel.box_size;
      this.numBoxes = newModel.num_boxes;
      this.isQuantized = newModel.isQuantized;
      if (this.isQuantized) {
        this.tensorType = Uint8Array;
        this.deQuantizedOutputBoxTensor = new Float32Array(this.numBoxes * this.boxSize);
        this.deQuantizedOutputClassScoresTensor = new Float32Array(this.numBoxes * this.numClasses);
      } else {
        this.tensorType = Float32Array;
      }
      
      // this.inputTensors = [new this.inputType(this.inputSize.reduce((a, b) => a * b))];
      this.outputBoxTensor = new this.tensorType(this.numBoxes * this.boxSize);
      this.outputClassScoresTensor = new this.tensorType(this.numBoxes * this.numClasses);
      this.outputTensors = this.prepareSsdOutputTensor(this.outputBoxTensor, this.outputClassScoresTensor);
    } else {
      this.anchors = newModel.anchors;
      // this.inputTensors = [new Float32Array(this.inputSize.reduce((a, b) => a * b))];
      this.outputTensors = [new Float32Array(this.outputSize)];
    }
    this.rawModel = null;

    let result = await this.loadModelAndLabels(this.modelFile, this.labelsFile);
    this.labels = result.text.split('\n');
    console.log(`labels: ${this.labels}`);
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
    const dummyInputTensor = new this.tensorType(this.inputSize.reduce((a, b) => a * b));
    let start = performance.now();
    result = await this.model.compute([dummyInputTensor], this.outputTensors);
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

  async predict(imageSource) {
    if (this.modelType === 'SSD') 
      return this.predictSSD(imageSource);
    else 
      return this.predictYolo(imageSource);
  }

  async predictSSD(imageSource) {
    if (!this.initialized) return;

    if (this.imageSource !== imageSource) {
      this.imageSource = imageSource;
      // initialize preprocessor
      this.preprocessor =
          new Preprocessor(this.imageSource, this.inputSize, this.tensorType, true, this.preOptions);
    }

    const inputTensor = await this.preprocessor.getFrame();
    const start = performance.now();
    await this.model.compute([inputTensor], this.outputTensors);
    const elapsed = performance.now() - start;
    console.log(`Inference time: ${elapsed.toFixed(2)} ms`);
    // console.log('outputBoxTensor', this.outputBoxTensor)
    // console.log('outputClassScoresTensor', this.outputClassScoresTensor)
    // let startDecode = performance.now();
    let outputBoxTensor, outputClassScoresTensor;
    if (this.isQuantized) {
      [outputBoxTensor, outputClassScoresTensor] = 
        this.deQuantizeOutputTensor(this.outputBoxTensor, this.outputClassScoresTensor, this.model._deQuantizeParams);
    } else {
      outputBoxTensor = this.outputBoxTensor;
      outputClassScoresTensor = this.outputClassScoresTensor;
    }
    decodeOutputBoxTensor({}, outputBoxTensor, this.anchors);
    // console.log(`Decode time: ${(performance.now() - startDecode).toFixed(2)} ms`);
    // let startNMS = performance.now();
    let [totalDetections, boxesList, scoresList, classesList] = NMS({}, outputBoxTensor, outputClassScoresTensor);
    boxesList = cropSSDBox(imageSource, totalDetections, boxesList, this.margin);
    // console.log(`NMS time: ${(performance.now() - startNMS).toFixed(2)} ms`);
    // let startVisual = performance.now();
    visualize(this.canvasShowElement, totalDetections, imageSource, boxesList, scoresList, classesList, this.labels);
    // console.log(`visual time: ${(performance.now() - startVisual).toFixed(2)} ms`);
    return {
      time: elapsed.toFixed(2)
    };
  }

  async predictYolo(imageSource) {
    if (!this.initialized) return;

    if (this.imageSource !== imageSource) {
      this.imageSource = imageSource;
      // initialize preprocessor
      this.preprocessor =
          new Preprocessor(imageSource, this.inputSize, this.tensorType, this.preOptions);
    }

    const inputTensor = await this.preprocessor.getFrame();
    const start = performance.now();
    await this.model.compute([inputTensor], this.outputTensors);
    const elapsed = performance.now() - start;
    console.log(`Inference time: ${elapsed.toFixed(2)} ms`);
    // let decodeStart = performance.now();
    let decode_out = decodeYOLOv2({nb_class: this.numClasses}, this.outputTensors[0], this.anchors);
    let boxes = getBoxes(decode_out, this.margin);
    // console.log(`Decode time: ${(performance.now() - decodeStart).toFixed(2)} ms`);
    // let drawStart = performance.now();
    drawBoxes(imageSource, this.canvasShowElement, boxes, this.labels);
    // console.log(`Draw time: ${(performance.now() - drawStart).toFixed(2)} ms`);
    return {
      time: elapsed.toFixed(2)
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

  prepareSsdOutputTensor(outputBoxTensor, outputClassScoresTensor) {
    let outputTensor = [];
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
      outputTensor[2 * i] = boxTensor;
      outputTensor[2 * i + 1] = classTensor;
      boxOffset += boxLen * outH[i];
      classOffset += classLen * outH[i];
    }
    return outputTensor;
  }

  deQuantizeOutputTensor(outputBoxTensor, outputClassScoresTensor, quantizedParams) {
    const outH = [1083, 600, 150, 54, 24, 6];
    const boxLen = 4;
    const classLen = 91;
    let boxOffset = 0;
    let classOffset = 0;
    let boxTensor, classTensor;
    let boxScale, boxZeroPoint, classScale, classZeroPoint;
    let dqBoxOffset = 0;
    let dqClassOffset = 0;
    for (let i = 0; i < 6; ++i) {
      boxTensor = outputBoxTensor.subarray(boxOffset, boxOffset + boxLen * outH[i]);
      classTensor = outputClassScoresTensor.subarray(classOffset, classOffset + classLen * outH[i]);
      boxScale = quantizedParams[2 * i].scale;
      boxZeroPoint = quantizedParams[2 * i].zeroPoint;
      classScale = quantizedParams[2 * i + 1].scale;
      classZeroPoint = quantizedParams[2 * i + 1].zeroPoint;
      for (let j = 0; j < boxTensor.length; ++j) {
        this.deQuantizedOutputBoxTensor[dqBoxOffset] = boxScale* (boxTensor[j] - boxZeroPoint);
        ++dqBoxOffset;
      }
      for (let j = 0; j < classTensor.length; ++j) {
        this.deQuantizedOutputClassScoresTensor[dqClassOffset] = classScale * (classTensor[j] - classZeroPoint);
        ++dqClassOffset;
      }
      boxOffset += boxLen * outH[i];
      classOffset += classLen * outH[i];
    }
    return [this.deQuantizedOutputBoxTensor, this.deQuantizedOutputClassScoresTensor];
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }
}