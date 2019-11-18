// Benchmark for Semantic Segmentation models
class SSBenchmark extends Benchmark {
  constructor(modelName, backend, iterations) {
    super(...arguments);
    this.modelInfoDict = getModelInfoDict(semanticSegmentationModels, modelName);
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
    let typedArrayIn = null;
    let typedArrayOut = null;
    if (this.isQuantized) {
      typedArrayIn = Uint8Array;
      typedArrayOut = Uint8Array;
    } else {
      typedArrayIn = Float32Array;
      typedArrayOut = Int32Array;
    }
    this.inputTensor = new typedArrayIn(this.modelInfoDict.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new typedArrayOut(this.modelInfoDict.outputSize.reduce((a, b) => a * b));
    this.inputSize = this.modelInfoDict.inputSize;
    this.outputSize = this.modelInfoDict.outputSize;
    height = this.inputSize[0];
    width = this.inputSize[1];
    let imWidth = imageElement.naturalWidth;
    let imHeight = imageElement.naturalHeight;
    // assume deeplab_out.width == deeplab_out.height
    let resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
    let dwidth = Math.floor(imWidth / resizeRatio);
    let dheight = Math.floor(imHeight / resizeRatio);
    canvasElement.setAttribute("width", width);
    canvasElement.setAttribute("height", height);
    let imageBytes = await loadImage(imageElement.src);
    let canvasContext = canvasElement.getContext('2d');
    canvasContext.drawImage(imageBytes, 0, 0, dwidth, dheight);
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
                                          [this.outputTensor]);
    console.log(`compute result: ${result}`);
  }

  async executeAsync() {
    let segCanvasElement = document.createElement('canvas');
    let results = [];
    for (let i = 0; i < this.iterations; i++) {
      this.onExecuteSingle(i);
      await new Promise(resolve => requestAnimationFrame(resolve));
      let tStart = performance.now();
      await this.executeSingleAsync();
      let elapsedTime = performance.now() - tStart;
      results.push(elapsedTime);
    }
    let imWidth = imageElement.width;
    let imHeight = imageElement.height;
    let resizeRatio = Math.max(Math.max(imWidth, imHeight) / this.inputSize[0], 1);
    let scaledWidth = Math.floor(imWidth / resizeRatio);
    let scaledHeight = Math.floor(imHeight / resizeRatio);
    let renderer = new Renderer(segCanvasElement);
    renderer.setup();
    renderer.uploadNewTexture(imageElement, [scaledWidth, scaledHeight]);
    renderer.drawOutputs({
      data: this.outputTensor,
      outputShape: this.outputSize,
      labels: this.labels,
    });
    let showCanvasContext = showCanvasElement.getContext('2d');
    showCanvasContext.drawImage(segCanvasElement, 0, 0, imWidth, imHeight);
    return results;
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
