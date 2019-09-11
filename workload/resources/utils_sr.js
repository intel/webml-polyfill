// Benchmark for Super Resolution models
class SRBenchmark extends Benchmark {
  constructor(modelName, backend, iterations) {
    super(...arguments);
    this.modelInfoDict = getModelInfoDict(superResolutionModels, modelName);
    this.model = null;
    this.inputTensor = null;
    this.outputTensor = null;
    this.isQuantized = null;
    this.outputCtx = showCanvasElement.getContext('2d');
  }

  async setInputOutput() {
    let channels = this.modelInfoDict.inputSize[2];
    let imageChannels = 4; // RGBA
    let [mean, offset] = [127.5, 1];
    this.isQuantized = this.modelInfoDict.isQuantized || false;
    let typedArray;
    if (this.isQuantized) {
      typedArray = Uint8Array;
    } else {
      typedArray = Float32Array;
    }
    this.inputTensor = new typedArray(this.modelInfoDict.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new typedArray(this.modelInfoDict.outputSize.reduce((a, b) => a * b));
    let inCanvas = document.createElement('canvas');
    inCanvas.width = this.modelInfoDict.inputSize[1];
    inCanvas.height = this.modelInfoDict.inputSize[0];
    let inCtx = inCanvas.getContext('2d');
    let imageBytes = await loadImage(imageElement.src);
    inCtx.drawImage(imageBytes, 0, 0, inCanvas.width, inCanvas.height);
    let pixels = inCtx.getImageData(0, 0, inCanvas.width, inCanvas.height).data;
    for (let y = 0; y < inCanvas.height; ++y) {
      for (let x = 0; x < inCanvas.width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y * inCanvas.width * imageChannels + x * imageChannels + c];
          this.inputTensor[y * inCanvas.width * channels + x * channels + c] = value / mean - offset;
        }
      }
    }
    showCanvasElement.setAttribute("height", imageElement.height * 2 + 5);
    this.outputCtx.drawImage(inCanvas, 0, 0, imageElement.width, imageElement.height);
  }

  /**
   * Setup model
   * @returns {Promise<void>}
   */
  async setupAsync() {
    await this.setInputOutput();
    let backend = this.backend.replace('WebNN', 'WebML');
    let loadResult = await loadModelAndLabels(this.modelInfoDict.modelFile);
    let flatBuffer = new flatbuffers.ByteBuffer(loadResult.bytes);
    let rawModel = tflite.Model.getRootAsModel(flatBuffer);
    let kwargs = {
      rawModel: rawModel,
      backend: backend,
      prefer: getPreferString()
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
    let results = [];
    for (let i = 0; i < this.iterations; i++) {
      this.onExecuteSingle(i);
      await new Promise(resolve => requestAnimationFrame(resolve));
      let tStart = performance.now();
      await this.executeSingleAsync();
      let elapsedTime = performance.now() - tStart;
      results.push(elapsedTime);
    }
    return results;
  }

  drawOutput() {
    let outCanvas = document.createElement('canvas');
    outCanvas.width = this.modelInfoDict.outputSize[1];
    outCanvas.height = this.modelInfoDict.outputSize[0];
    let [mean, offset] = [127.5, 1];
    let bytes = new Uint8ClampedArray(outCanvas.width * outCanvas.height * 4);
    for (let i = 0; i < outCanvas.height * outCanvas.width; ++i) {
      let j = i * 4;
      let r, g, b, a;
      r = (this.outputTensor[i * 3] + offset) * mean;
      g = (this.outputTensor[i * 3 + 1] + offset) * mean;
      b = (this.outputTensor[i * 3 + 2] + offset) * mean;
      a = 255;
      bytes[j + 0] = Math.round(r);
      bytes[j + 1] = Math.round(g);
      bytes[j + 2] = Math.round(b);
      bytes[j + 3] = Math.round(a);
    }
    let imageData = new ImageData(bytes, outCanvas.width, outCanvas.height);
    let outCtx = outCanvas.getContext('2d');
    outCtx.putImageData(imageData, 0, 0);
    this.outputCtx.drawImage(outCanvas, 0, imageElement.height + 5, imageElement.width, imageElement.height);
  }

  handleResults(results) {
    this.drawOutput();
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
