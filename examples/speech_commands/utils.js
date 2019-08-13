class Utils {
  constructor(audio) {
    this.rawModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;
    this.modelFile;
    this.labelsFile;
    this.inputSize;
    this.outputSize;
    this.sampleRate;
    this.preOptions;
    this.postOptions;
    this.audioContext = audio;
    this.updateProgress;
    this.backend = '';
    this.prefer = '';
    this.initialized = false;
    this.loaded = false;
    this.resolveGetRequiredOps = null;
    this.outstandingRequest = null;
  }

  async loadModel(model) {
    if (this.loaded && this.modelFile === model.modelFile) {
      return 'LOADED';
    }
    // reset all states
    this.loaded = this.initialized = false;
    this.backend = this.prefer = '';

    // set new model params
    this.inputSize = model.inputSize;
    this.outputSize = model.outputSize;
    this.sampleRate = model.sampleRate;
    this.modelFile = model.modelFile;
    this.labelsFile = model.labelsFile;
    this.preOptions = model.preOptions || {};
    this.postOptions = model.postOptions || {};
    this.isQuantized = model.isQuantized || false;
    let typedArray;
    if (this.isQuantized) {
      typedArray = Uint8Array;
    } else {
      typedArray = Float32Array;
    }
    this.inputTensor = new typedArray(this.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new typedArray(this.outputSize.reduce((a, b) => a * b));

    let result = await this.loadModelAndLabels(this.modelFile, this.labelsFile);
    this.labels = result.text.split('\n');
    console.log(`labels: ${this.labels}`);

    switch (this.modelFile.split('.').pop()) {
      case 'tflite':
        let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
        this.rawModel = tflite.Model.getRootAsModel(flatBuffer);
        this.rawModel._rawFormat = 'TFLITE';
        printTfLiteModel(this.rawModel);
        break;
      case 'onnx':
        let err = onnx.ModelProto.verify(result.bytes);
        if (err) {
          throw new Error(`Invalid model ${err}`);
        }
        this.rawModel = onnx.ModelProto.decode(result.bytes);
        this.rawModel._rawFormat = 'ONNX';
        printOnnxModel(this.rawModel);
        break;
      case 'bin':
        const networkFile = this.modelFile.replace(/bin$/, 'xml');
        const networkText = await this.loadUrl(networkFile, false, false);
        const weightsBuffer = result.bytes.buffer;
        this.rawModel = new OpenVINOModel(networkText, weightsBuffer);
        this.rawModel._rawFormat = 'OPENVINO';
        break;
      default:
        throw new Error('Unrecognized model format');
    }
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
    let configs = {
      rawModel: this.rawModel,
      backend: backend,
      prefer: prefer,
      softmax: this.postOptions.softmax || false,
    };
    switch (this.rawModel._rawFormat) {
      case 'TFLITE':
        this.model = new TFliteModelImporter(configs);
        break;
      case 'ONNX':
        this.model = new OnnxModelImporter(configs);
        break;
      case 'OPENVINO':
        this.model = new OpenVINOModelImporter(configs);
        break;
    }
    let result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute([this.inputTensor], [this.outputTensor]);
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

  async predict(mediaElement) {
    if (!this.initialized) return;
    await this.prepareInputTensor(this.inputTensor, mediaElement);
    let start = performance.now();
    await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    return {
      time: elapsed.toFixed(2),
      classes: this.getTopClasses(this.outputTensor, this.labels, 3, this.model._deQuantizeParams)
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

  async prepareInputTensor(tensor, audio) {
    let request = new Request(audio.src);
    let response = await fetch(request);
    let audioFileData = await response.arrayBuffer();
    let audioDecodeData = await this.audioContext.decodeAudioData(audioFileData);
    let audioPCMData = audioDecodeData.getChannelData(0);
    let inputTensor = this.downsampleAudioBuffer(audioPCMData, this.sampleRate);
    if(inputTensor.length >= this.inputSize) {
      for(let i = 0; i < this.inputSize; i++) {
        tensor[i] = inputTensor[i];
      }
    } else {
      for(let i = 0; i < inputTensor.length; i++) {
        tensor[i] = inputTensor[i];
      }
    }
  }

  downsampleAudioBuffer(buffer, rate) {
    let sampleRate = this.audioContext.sampleRate;

    if (rate == sampleRate) {
        return buffer;
    }
    if (rate > sampleRate) {
        throw "downsampling rate show be smaller than original sample rate";
    }
    var sampleRateRatio = Math.round(sampleRate / rate);
    var newLength = Math.round(buffer.length / sampleRateRatio);
    var result = new Float32Array(newLength);
    var offsetResult = 0;
    var offsetBuffer = 0;
    while (offsetResult < result.length) {
        var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        var accum = 0,
            count = 0;
        for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  getTopClasses(tensor, labels, k = 5, deQuantizeParams) {
    let outputTensor = this.prepareOutputTensor(tensor);
    let probs = Array.from(outputTensor);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
      if (a[0] === b[0]) {return 0;}
      return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    let classes = [];
    for (let i = 0; i < k; ++i) {
      let prob;
      if (this.isQuantized) {
        prob = deQuantizeParams[0].scale * (sorted[i][0] - deQuantizeParams[0].zeroPoint);
      } else {
        prob = sorted[i][0];
      }
      let index = sorted[i][1];
      let c = {
        label: labels[index],
        prob: (prob * 100).toFixed(2)
      }
      classes.push(c);
    }
    return classes;
  }

  prepareOutputTensor(tensor) {
    let tensorRuduceMax = Math.max.apply(null, tensor);
    let tensorSub = new Array(tensor.length);
    let tensorExp = new Array(tensor.length);
    let outputTensor = new Array(tensor.length);
    for(let i = 0; i < tensor.length; i++) {
      tensorSub[i] = tensor[i] - tensorRuduceMax;
    }
    for(let i = 0; i < tensor.length; i++) {
      tensorExp[i] = Math.exp(tensorSub[i]);
    }
    let tensorSum = eval(tensorExp.join("+"));
    for(let i = 0; i < tensor.length; i++) {
      outputTensor[i] = tensorExp[i] / tensorSum;
    }
    return outputTensor;
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

}