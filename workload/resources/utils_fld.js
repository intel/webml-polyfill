// Benchmark for Facial Landmark Detection models
class FLDBenchmark extends Benchmark {
  constructor(modelName, backend, iterations) {
    super(...arguments);
    this.faceDetector = null;
    this.modelName = modelName;
    this.modelInfoDict = getModelInfoDict(facialLandmarkDetectionModels, 'SimpleCNN (TFlite)');
    this.model = null;
    this.inputTensor = null;
    this.inputSize = null;
    this.outputTensor = null;
    this.outputSize = null;
  }

  async setupFaceDetector() {
    let inputCanvas = document.createElement('canvas');
    let model = faceDetectionModels.filter(f => f.modelName == this.modelName);
    inputCanvas.setAttribute("width", model[0].inputSize[1]);
    inputCanvas.setAttribute("height", model[0].inputSize[0]);
    let modelFile = model[0].modelFile;
    if (!modelFile.toLowerCase().startsWith("https://") && !modelFile.toLowerCase().startsWith("http://")) {
       model[0].modelFile = '../examples/util/' + model[0].modelFile;
    }
    this.faceDetector = new FaceDetecor(inputCanvas);
    await this.faceDetector.loadModel(model[0]);
    await this.faceDetector.init(this.backend.replace('WebNN', 'WebML'), preferSelect.value);
    model[0].modelFile = modelFile;
  }

  async getFaceDetectResult() {
    let detectResult = await this.faceDetector.getFaceBoxes(imageElement);
    return detectResult;
  }

  async setInputOutput(box) {
    let inputCanvas = document.createElement('canvas');
    let width = this.modelInfoDict.inputSize[1];
    let height = this.modelInfoDict.inputSize[0];
    const channels = this.modelInfoDict.inputSize[2];
    const imageChannels = 4; // RGBA
    const preOptions = this.modelInfoDict.preOptions || {};
    const mean = preOptions.mean || [0, 0, 0, 0];
    const std = preOptions.std  || [1, 1, 1, 1];
    const norm = preOptions.norm || false;
    const channelScheme = preOptions.channelScheme || 'RGB';
    let typedArray = Float32Array;
    this.inputTensor = new typedArray(this.modelInfoDict.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new typedArray(this.modelInfoDict.outputSize);
    inputCanvas.setAttribute("width", width);
    inputCanvas.setAttribute("height", height);
    let canvasContext = inputCanvas.getContext('2d');
    canvasContext.drawImage(imageElement, box[0], box[2],
                            box[1]-box[0], box[3]-box[2], 0, 0,
                            inputCanvas.width,
                            inputCanvas.height);
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
    await this.setupFaceDetector();
    let backend = this.backend.replace('WebNN', 'WebML');
    let loadResult = await loadModelAndLabels(this.modelInfoDict.modelFile);
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

  async executeSingleAsync() {
    let keyPoints = [];
    let result = null;
    let detectResult = await this.getFaceDetectResult();
    let faceBoxes = detectResult.boxes;
    for (let i = 0; i < faceBoxes.length; ++i) {
      this.setInputOutput(faceBoxes[i]);
      result = await this.model.compute([this.inputTensor], [this.outputTensor]);
      let outputTensor = this.outputTensor;
      keyPoints.push(outputTensor);
    }
    console.log(`compute status: ${result}`);
    return {faceBoxes: faceBoxes, keyPoints: keyPoints};
  }

  async executeAsync() {
    let results = [];
    let exeResult = null;
    for (let i = 0; i < this.iterations; i++) {
      this.onExecuteSingle(i);
      await new Promise(resolve => requestAnimationFrame(resolve));
      let tStart = performance.now();
      exeResult = await this.executeSingleAsync();
      let elapsedTime = performance.now() - tStart;
      results.push(elapsedTime);
    }
    this.drawFaceBoxes(imageElement, showCanvasElement, exeResult.faceBoxes);
    this.drawKeyPoints(imageElement, showCanvasElement, exeResult.keyPoints, exeResult.faceBoxes);
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
      this.faceDetector.model._compilation._preparedModel._deleteAll();
    }
  }

  drawFaceBoxes(image, canvas, face_boxes) {
    canvas.width = image.width / image.height * canvas.height;
    // drawImage
    let ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  
    // drawFaceBox
    face_boxes.forEach(box => {
      let xmin = box[0] / image.height * canvas.height;
      let xmax = box[1] / image.height * canvas.height;
      let ymin = box[2] / image.height * canvas.height;
      let ymax = box[3] / image.height * canvas.height;
      let prob = box[4];
      ctx.strokeStyle = "#009bea";
      ctx.fillStyle = "#009bea";
      ctx.lineWidth = 3;
      ctx.strokeRect(xmin, ymin, xmax-xmin, ymax-ymin);
      ctx.font = "18px Arial";
      let text = `${prob.toFixed(2)}`;
      let width = ctx.measureText(text).width;
      if (xmin >= 2 && ymin >= parseInt(ctx.font, 10)) {
        ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10), width + 4, parseInt(ctx.font, 10));
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin, ymin - 3);
      } else {
        ctx.fillRect(xmin + 2, ymin , width + 4,  parseInt(ctx.font, 10));
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin + 2, ymin + 15);
      }
    });
  }

  drawKeyPoints(image, canvas, Keypoints, boxes) {
    let keypoints = null;
    let ctx = canvas.getContext('2d');
    boxes.forEach((box, n) => {
      keypoints = Keypoints[n];
      for (let i = 0; i < 136; i = i + 2) {
        // decode keypoints
        let x = ((box[1] - box[0]) * keypoints[i] + box[0]) / image.width * canvas.width;
        let y = ((box[3] - box[2]) * keypoints[i + 1] + box[2]) / image.height * canvas.height;
        // draw keypoints
        ctx.beginPath();
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
        ctx.closePath();
      }
    });
  }
}
