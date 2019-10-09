// Benchmark for Emotion Analysis Detection models
class EABenchmark extends Benchmark {
  constructor(modelName, backend, iterations) {
    super(...arguments);
    this.faceDetector = null;
    this.modelName = modelName;
    this.modelInfoDict = getModelInfoDict(emotionAnalysisModels, 'Simple CNN 7 (TFlite)');
    this.model = null;
    this.inputTensor = null;
    this.inputSize = null;
    this.outputTensor = null;
    this.outputSize = null;
    this.outputBoxTensor = null;
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
    this.inputTensor = new Float32Array(this.modelInfoDict.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new Float32Array(this.modelInfoDict.outputSize);
    inputCanvas.setAttribute("width", width);
    inputCanvas.setAttribute("height", height);
    let canvasContext = inputCanvas.getContext('2d');
    canvasContext.drawImage(imageElement, box[0], box[2],
                            box[1] - box[0], box[3] - box[2], 0, 0,
                            width, height);
    let pixels = canvasContext.getImageData(0, 0, width, height).data;
    if (norm) {
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let index = y * width * imageChannels + x * imageChannels + c;
            let value = (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3 / 255;
            this.inputTensor[y * width * channels + x * channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    }
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
    let classes = this.getTopClasses(exeResult.keyPoints, this.modelInfoDict.labels, 1);
    this.drawFaceBoxes(imageElement, showCanvasElement, exeResult.faceBoxes, classes);
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

  drawFaceBoxes(image, canvas, face_boxes, classes) {
    canvas.width = image.width / image.height * canvas.height;
    // drawImage
    let ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    // drawFaceBox
    face_boxes.forEach((box, i) => {
      let xmin = box[0] / image.height * canvas.height;
      let xmax = box[1] / image.height * canvas.height;
      let ymin = box[2] / image.height * canvas.height;
      let ymax = box[3] / image.height * canvas.height;
      ctx.strokeStyle = "#009bea";
      ctx.fillStyle = "#009bea";
      ctx.lineWidth = 3;
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
      ctx.font = "20px Arial";
      let prob = classes[i].prob;
      let label = classes[i].label;
      let text = `${label}:${prob}`;
      let width = ctx.measureText(text).width;
      if (xmin >= 2 && ymin >= parseInt(ctx.font, 10)) {
        ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10), width + 4, parseInt(ctx.font, 10));
        ctx.fillStyle = "white";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin, ymin - 3);
      } else {
        ctx.fillRect(xmin + 2, ymin, width + 4, parseInt(ctx.font, 10));
        ctx.fillStyle = "white";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin + 2, ymin + 15);
      }
    });
  }

  getTopClasses(tensors, labels, k = 3) {
    let classes = [];
    tensors.forEach(tensor => {
      let probs = Array.from(tensor);
      let indexes = probs.map((prob, index) => [prob, index]);
      let sorted = indexes.sort((a, b) => {
        if (a[0] === b[0]) {return 0;}
        return a[0] < b[0] ? -1 : 1;
      });
      sorted.reverse();
      for (let i = 0; i < k; ++i) {
        let prob = sorted[i][0];
        let index = sorted[i][1];
        let c = {
          label: labels[index],
          prob: (prob * 100).toFixed(2)
        }
        classes.push(c);
      }
    });
    return classes;
  }
}
