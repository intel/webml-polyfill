// Benchmark for Image Classification models
class ICBenchmark extends Benchmark {
  constructor(modelName, backend, iterations) {
    super(...arguments);
    this.modelInfoDict = getModelInfoDict(imageClassificationModels, modelName);
    this.model = null;
    this.labels = null;
    this.inputTensor = null;
    this.inputSize = null;
    this.outputTensor = null;
    this.outputSize = null;
    this.isQuantized = false;
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
    this.inputTensor = new typedArray(this.modelInfoDict.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new typedArray(this.modelInfoDict.outputSize);
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
    const modelFormat = this.modelInfoDict.format;
    let loadResult = await loadModelAndLabels(this.modelInfoDict.modelFile, this.modelInfoDict.labelsFile);
    this.labels = loadResult.text.split('\n');
    let importerClass;
    let rawModel;
    switch (modelFormat) {
      case 'TFLite':
        let flatBuffer = new flatbuffers.ByteBuffer(loadResult.bytes);
        rawModel = tflite.Model.getRootAsModel(flatBuffer);
        importerClass = TFliteModelImporter;
        break;
      case 'ONNX':
        let err = onnx.ModelProto.verify(loadResult.bytes);
        if (err) {
          throw new Error(`Invalid model ${err}`);
        }
        rawModel = onnx.ModelProto.decode(loadResult.bytes);
        importerClass = OnnxModelImporter;
        break;
      case 'OpenVINO':
        const networkFile = this.modelInfoDict.modelFile.replace(/bin$/, 'xml');
        let networkText;

        if (networkFile.toLowerCase().startsWith("https://") || networkFile.toLowerCase().startsWith("http://")) {
          networkText = await loadUrl(networkFile, false);
        } else {
          networkText = await loadUrl('../examples/util/' + networkFile, false);
        }

        const weightsBuffer = loadResult.bytes.buffer;
        rawModel = new OpenVINOModel(networkText, weightsBuffer);
        importerClass = OpenVINOModelImporter;
        break;
    }
    let postOptions = this.modelInfoDict.postOptions || {};
    let kwargs = {
      rawModel: rawModel,
      backend: backend,
      prefer: getPreferString(),
      softmax: postOptions.softmax || false,
    };
    this.model = new importerClass(kwargs);
    supportedOps = getSelectedOps();
    await this.model.createCompiledModel();
  }

  /**
   * Execute model
   * @returns {Promise<void>}
   */
  async executeSingleAsync() {
    let result = await this.model.compute([this.inputTensor],
                                          [this.outputTensor]);
    console.log(`compute status: ${result}`);
  }

  async executeAsync() {
    let results = [];
    for (let i = 0; i < this.iterations; i++) {
      this.onExecuteSingle(i);
      await new Promise(resolve => requestAnimationFrame(resolve));
      let tStart = performance.now();
      await this.executeSingleAsync();
      let elapsedTime = performance.now() - tStart;
      this.printPredictResult();
      results.push(elapsedTime);
    }
    return results;
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
    let prob;
    for (let i = 0; i < 3; ++i) {
      if (this.isQuantized) {
        prob = this.model._deQuantizeParams[0].scale * (sorted[i][0] - this.model._deQuantizeParams[0].zeroPoint);
      } else {
        prob = sorted[i][0];
      }
      let index = sorted[i][1];
      console.log(`label: ${this.labels[index]}, probability: ${(prob * 100).toFixed(2)}%`);
    }
  }

  handleResults(results) {
    let profilingResults = null;
    if (this.backend !== 'WebNN') {
      profilingResults = this.model._compilation._preparedModel.dumpProfilingResults();
    }
    return {
      "computeResults": summarize(results),
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
