// Benchmark for Object Detection models
class ODBenchmark extends Benchmark {
  constructor(modelName, backend, iterations) {
    super(...arguments);
    this.modelInfoDict = getModelInfoDict(objectDetectionModels, modelName);
    this.model = null;
    this.labels = null;
    this.inputTensor = null;
    this.inputSize = null;
    this.outputTensor = null;
    this.outputSize = null;
    this.outputBoxTensor = null;
    this.outputClassScoresTensor = null;
    this.deQuantizedOutputBoxTensor = null;
    this.deQuantizedOutputClassScoresTensor = null;
    this.deQuantizeParams = null;
    this.anchors = null;
    this.modelType = null;
    this.modelMargin = null;
    this.numClasses = 0;
    this.isQuantized = false;
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

  async setInputOutput() {
    let canvasElement = document.createElement('canvas');
    let width = this.modelInfoDict.inputSize[1];
    let height = this.modelInfoDict.inputSize[0];
    const channels = this.modelInfoDict.inputSize[2];
    const preOptions = this.modelInfoDict.preOptions || {};
    const mean = preOptions.mean || [0, 0, 0, 0];
    const std = preOptions.std  || [1, 1, 1, 1];
    const norm = preOptions.norm || false;
    const channelScheme = preOptions.channelScheme || 'RGB';
    const imageChannels = 4; // RGBA
    this.isQuantized = this.modelInfoDict.isQuantized || false;
    let typedArray;
    if (this.isQuantized) {
      typedArray = Uint8Array;
    } else {
      typedArray = Float32Array;
    }
    this.outputTensor = [];
    this.modelType = this.modelInfoDict.type;
    this.modelMargin = this.modelInfoDict.margin;
    this.numClasses = this.modelInfoDict.num_classes;
    if (this.modelInfoDict.type === 'SSD') {
      if (this.isQuantized) {
        this.deQuantizedOutputBoxTensor = new Float32Array(this.modelInfoDict.num_boxes * this.modelInfoDict.box_size);
        this.deQuantizedOutputClassScoresTensor = new Float32Array(this.modelInfoDict.num_boxes * this.modelInfoDict.num_classes);
      }
      this.inputTensor = new typedArray(this.modelInfoDict.inputSize.reduce((a, b) => a * b));
      this.outputBoxTensor = new typedArray(this.modelInfoDict.num_boxes * this.modelInfoDict.box_size);
      this.outputClassScoresTensor = new typedArray(this.modelInfoDict.num_boxes * this.modelInfoDict.num_classes);
      this.prepareoutputTensor(this.outputBoxTensor, this.outputClassScoresTensor);
      this.anchors = generateAnchors({});
    } else {
      // YOLO
      this.inputTensor = new typedArray(this.modelInfoDict.inputSize.reduce((a, b) => a * b));
      this.anchors = this.modelInfoDict.anchors;
      this.outputTensor = [new typedArray(this.modelInfoDict.outputSize)]
    }
    canvasElement.setAttribute("width", width);
    canvasElement.setAttribute("height", height);
    let canvasContext = canvasElement.getContext('2d');
    canvasContext.drawImage(imageElement, 0, 0, width, height);
    let pixels = canvasContext.getImageData(0, 0, width, height).data;
    if (norm) {
      pixels = new Float32Array(pixels).map(p => p / 255);
    }
    setInputTensor(pixels, imageChannels, height, width, channels,
                   channelScheme, mean, std, this.inputTensor);
  }

  /**
   * Setup model
   * @returns {Promise<void>}
   */
  async setupAsync() {
    await this.setInputOutput();
    let backend = this.backend.replace('WebNN', 'WebML');
    let loadResult = await loadModelAndLabels(this.modelInfoDict.modelFile, this.modelInfoDict.labelsFile);
    this.labels = loadResult.text.split('\n');
    let flatBuffer = new flatbuffers.ByteBuffer(loadResult.bytes);
    let rawModel = tflite.Model.getRootAsModel(flatBuffer);
    let postOptions = this.modelInfoDict.postOptions || {};
    let kwargs = {
      rawModel: rawModel,
      backend: backend,
      prefer: getPreferString(),
      softmax: postOptions.softmax || false,
    };
    this.model = new TFliteModelImporter(kwargs);
    supportedOps = getSelectedOps();
    await this.model.createCompiledModel();
  }

  /**
   * Execute model
   * @returns {Promise<void>}
   */
  async executeSingleAsync() {
    let result = await this.model.compute([this.inputTensor],
                                          this.outputTensor);
    console.log(`compute status: ${result}`);
  }

  async executeAsync() {
    let computeResults = [];
    let decodeResults = [];
    if (this.modelType === 'SSD') {
      let outputBoxTensor, outputClassScoresTensor;
      for (let i = 0; i < this.iterations; i++) {
        this.onExecuteSingle(i);
        await new Promise(resolve => requestAnimationFrame(resolve));
        let tStart = performance.now();
        await this.executeSingleAsync();
        let elapsedTime = performance.now() - tStart;
        computeResults.push(elapsedTime);
        if (this.isQuantized) {
          [outputBoxTensor, outputClassScoresTensor] = this.deQuantizeOutputTensor(this.outputBoxTensor, this.outputClassScoresTensor, this.model._deQuantizeParams);
        } else {
          outputBoxTensor = this.outputBoxTensor;
          outputClassScoresTensor = this.outputClassScoresTensor;
        }
        let dstart = performance.now();
        decodeOutputBoxTensor({}, outputBoxTensor, this.anchors);
        let decodeTime = performance.now() - dstart;
        console.log("Decode time:" + decodeTime);
        decodeResults.push(decodeTime);
      }
      let [totalDetections, boxesList, scoresList, classesList] = NMS({}, outputBoxTensor, outputClassScoresTensor);
      showCanvasElement.setAttribute("width", imageElement.width);
      showCanvasElement.setAttribute("height", imageElement.height);
      visualize(showCanvasElement, totalDetections, imageElement, boxesList, scoresList, classesList, this.labels);
    } else {
      let decode_out;
      for (let i = 0; i < this.iterations; i++) {
        this.onExecuteSingle(i);
        await new Promise(resolve => requestAnimationFrame(resolve));
        let tStart = performance.now();
        await this.executeSingleAsync();
        let elapsedTime = performance.now() - tStart;
        computeResults.push(elapsedTime);
        let dstart = performance.now();
        decode_out = decodeYOLOv2({nb_class: this.numClasses}, this.outputTensor[0], this.anchors);
        let decodeTime = performance.now() - dstart;
        console.log("Decode time:" + decodeTime);
        decodeResults.push(decodeTime);
      }
      let boxes = getBoxes(decode_out, this.modelMargin);
      showCanvasElement.setAttribute("width", imageElement.width);
      showCanvasElement.setAttribute("height", imageElement.height);
      drawBoxes(imageElement, showCanvasElement, boxes, this.labels);
    }
    return {
      "computeResults": computeResults,
      "decodeResults": decodeResults
    };
  }

  handleResults(results) {
    let profilingResults = null;
    if (this.backend !== 'WebNN') {
      profilingResults = this.model._compilation._preparedModel.dumpProfilingResults();
    }
    return {
      "computeResults": summarize(results.computeResults),
      "decodeResults": summarize(results.decodeResults),
      "profilingResults": summarizeProf(profilingResults)
    };
  }

  /**
   * Finalize
   * @returns {Promise<void>}
   */
  finalize() {
    if (this.backend !== 'WebNN') {
      // explictly release memory of GPU texture or WASM heap
      this.model._compilation._preparedModel._deleteAll();
    }
  }
}
