let pnConfigDic = null;

// Benchmark for Skeleton Detection models
class SDBenchmark extends Benchmark {
  constructor(modelName, backend, iterations) {
    super(...arguments);
    this.modelInfoDict = getModelInfoDict(humanPoseEstimationModels, modelName);
    this.model = null;
    this.labels = null;
    this.inputTensor = null;
    this.inputSize = null;
    this.modelVersion = null;
    this.outputStride = null;
    this.scaleFactor = null;
    this.minScore = null;
    this.scaleWidth = null;
    this.scaleHeight = null;
    this.scaleInputSize = null;
    this.heatmapTensor = null;
    this.offsetTensor  = null;
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
    if (pnConfigDic === null) {
      // Read modelVersion outputStride scaleFactor minScore from json file
      let posenetConfigURL = './resources/posenetConfig.json';
      let pnConfigText = await loadUrl(posenetConfigURL);
      pnConfigDic = JSON.parse(pnConfigText);
    }
    this.modelVersion = Number(pnConfigDic.modelVersion);
    this.outputStride = Number(pnConfigDic.outputStride);
    this.scaleFactor = Number(pnConfigDic.scaleFactor);
    this.minScore = Number(pnConfigDic.minScore);
    this.scaleWidth = getValidResolution(this.scaleFactor, width, this.outputStride);
    this.scaleHeight = getValidResolution(this.scaleFactor, height, this.outputStride);
    this.inputTensor = new typedArray(this.scaleWidth * this.scaleHeight * channels);
    this.scaleInputSize = [1, this.scaleWidth, this.scaleHeight, channels];
    let HEATMAP_TENSOR_SIZE;
    if ((this.modelVersion == 0.75 || this.modelVersion == 0.5) && this.outputStride == 32) {
      HEATMAP_TENSOR_SIZE = product(toHeatmapsize(this.scaleInputSize, 16));
    } else {
      HEATMAP_TENSOR_SIZE = product(toHeatmapsize(this.scaleInputSize, this.outputStride));
    }
    let OFFSET_TENSOR_SIZE = HEATMAP_TENSOR_SIZE * 2;
    this.heatmapTensor = new typedArray(HEATMAP_TENSOR_SIZE);
    this.offsetTensor = new typedArray(OFFSET_TENSOR_SIZE);
    width = this.scaleWidth;
    height = this.scaleHeight;
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
    let modelArch = ModelArch.get(this.modelVersion);
    let smType = 'Singleperson';
    let cacheMap = new Map();
    let useAtrousConv = false; // Default false, NNAPI and BNNS don't support AtrousConv
    this.model = new PoseNet(modelArch, {'version': this.modelVersion, 'adjustPath': true,},
                             useAtrousConv, this.outputStride, this.scaleInputSize,
                             smType, cacheMap, backend, getPreferString());
    supportedOps = getSelectedOps();
    await this.model.createCompiledModel();
  }

  /**
   * Execute model
   * @returns {Promise<void>}
   */
  async executeSingleAsync() {
    let result = await this.model.computeSinglePose(this.inputTensor,
                                                    this.heatmapTensor,
                                                    this.offsetTensor);
    console.log(`compute result: ${result}`);
  }

  async executeAsync() {
    let computeResults = [];
    let decodeResults = [];
    let singlePose = null;
    for (let i = 0; i < this.iterations; i++) {
      this.onExecuteSingle(i);
      await new Promise(resolve => requestAnimationFrame(resolve));
      let tStart = performance.now();
      await this.executeSingleAsync();
      let elapsedTime = performance.now() - tStart;
      computeResults.push(elapsedTime);
      let dstart = performance.now();
      singlePose = decodeSinglepose(sigmoid(this.heatmapTensor),
                                    this.offsetTensor,
                                    toHeatmapsize(this.scaleInputSize,
                                                  this.outputStride),
                                    this.outputStride);
      let decodeTime = performance.now() - dstart;
      console.log("Decode time:" + decodeTime);
      decodeResults.push(decodeTime);
    }
    // draw canvas by last result
    let ctx = showCanvasElement.getContext('2d');
    let scaleX = showCanvasElement.width / this.scaleWidth;
    let scaleY = showCanvasElement.height / this.scaleHeight;
    singlePose.forEach((pose) => {
      if (pose.score >= this.minScore) {
        drawKeypoints(pose.keypoints, this.minScore, ctx, scaleX, scaleY);
        drawSkeleton(pose.keypoints, this.minScore, ctx, scaleX, scaleY);
      }
    });
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
