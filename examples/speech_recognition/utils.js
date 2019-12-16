class Utils {
  constructor(audio) {
    this.rawModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.inputTensorC;
    this.inputTensorH;
    this.outputTensor;
    this.outputTensorC;
    this.outputTensorH;
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

    this.inputTensorC = new Float32Array(2048);
    this.inputTensorH = new Float32Array(2048);
    this.outputTensorC = new Float32Array(2048);
    this.outputTensorH = new Float32Array(2048);

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
    let text = '';
    let timeArray = [];
    let decodeTimeArray = [];
    let result = [];

    this.inputTensorC.fill(0);
    this.inputTensorH.fill(0);

    console.log("~~~~~~Backend~~~~~~~ ", tf.getBackend());
    
    if (!this.initialized) return;
    let preStart = performance.now();
    let preparedTensor = await this.prepareInputTensor(mediaElement);
    let preElapsed = (performance.now() - preStart) / preparedTensor.length;

    for(let i=0; i<preparedTensor.length; i++) {
      this.inputTensor = preparedTensor[i];
      this.inputTensorC = this.outputTensorC;
      this.inputTensorH = this.outputTensorH;

      let start = performance.now();
      await this.model.compute([this.inputTensor, this.inputTensorC, this.inputTensorH], 
                               [this.outputTensor, this.outputTensorC, this.outputTensorH]);
      let elapsed = performance.now() - start;
      timeArray.push(elapsed);

      let logist = this.splitArr(this.outputTensor, 29);

      await tf.setBackend('cpu');
      console.log('*** Set backend to ~~~CPU~~~ and get backend:', tf.getBackend());
      let decodeStart = performance.now();
      for(let i=0; i<logist.length; i++) {
        let t = this.labels[tf.argMax(logist[i]).dataSync()];
        if(t !== text[text.length-1]) {
          text += t;
        }
      }
      let decodeElapsed = performance.now() - decodeStart;
      await tf.setBackend('webgl');
      console.log('*** Set backend to ~~~WebGL~~~ and get backend:', tf.getBackend());
      decodeTimeArray.push(decodeElapsed);
      console.log('round', i, 'inference:', elapsed, 'decode:', decodeElapsed);
    }
    console.log("Preprocess time:", preElapsed)
    console.log("Inference time:", eval(timeArray.join("+")) / timeArray.length);
    console.log("Decode time:", eval(decodeTimeArray.join("+")) / decodeTimeArray.length);
    console.log("Decode text:", text);
    let time = eval(timeArray.join("+")) / timeArray.length;
    result.push({label: text, prob: (0 * 100).toFixed(2)});

    return {
      time: time.toFixed(2),
      classes: result
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

  async prepareInputTensor(audio) {
    let request = new Request(audio.src);
    let response = await fetch(request);
    let audioFileData = await response.arrayBuffer();
    let audioDecodeData = await this.audioContext.decodeAudioData(audioFileData);
    let audioPCMData = audioDecodeData.getChannelData(0);

    let start = performance.now();
    let audioMfccs = this.getAudioMfccs(audioPCMData);
    let elapsed = performance.now() - start;
    console.log("---MFCCS time---", elapsed);
    let inputTensor = this.create_overlapping_windows(audioMfccs);

    return inputTensor;
  }

  getAudioMfccs(pcm) {
    let mfccsLenPtr = Module._malloc(4);
    let pcmPtr = Module._malloc(8 * pcm.length);

    for(let i=0; i<pcm.length; i++) {
      Module.HEAPF64[pcmPtr/8 + i] = pcm[i];
    }

    let tfMfccs = Module.cwrap('tf_mfccs', 'number', ['number', 'number', 'number']);
    let mfccsPtr = tfMfccs(pcmPtr, pcm.length, mfccsLenPtr);
    let mfccsLen = Module.HEAP32[mfccsLenPtr >> 2];
    let audioMfccs = [mfccsLen];

    for(let i=0; i<mfccsLen; i++) {
      audioMfccs[i] = Module.HEAPF64[(mfccsPtr >> 3) + i];
    }
    Module._free(pcmPtr, mfccsLenPtr);

    return audioMfccs;
  }

  create_overlapping_windows(batch_x) {
    let batch_size = 1;
    let window_width = 2 * 9 + 1;
    let num_channels = 26;
    let batchs = batch_x.length / num_channels;
  
    batch_x = tf.reshape(batch_x, [batchs, num_channels]);
    batch_x = tf.expandDims(batch_x, 0);
  
    // Create a constant convolution filter using an identity matrix, so that the
    // convolution returns patches of the input tensor as is, and we can create
    // overlapping windows over the MFCCs.
    let eye_tensor = tf.eye(window_width * num_channels);
    let eye_filter = tf.reshape(eye_tensor, [window_width, num_channels, window_width * num_channels])
  
    // Create overlapping windows
    batch_x = tf.conv1d(batch_x, eye_filter, 1, 'same');
  
    // Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels]);
    let batch_num = Math.floor(batch_x.shape[1] / 16);
    let slice_num = batch_num * 16;

    batch_x = tf.slice(batch_x, [0, 0, 0, 0], [batch_size, slice_num, window_width, num_channels]);
    let batch_xs = tf.split(batch_x, batch_num, 1);

    let result = [];
    for(let i=0; i<batch_num; i++) {
      result.push(batch_xs[i].dataSync())
    }

    return result;
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

  splitArr(data, senArrLen) {
    let dataLen = data.length;
    let arrOuterLen = dataLen % senArrLen === 0 ? dataLen / senArrLen : parseInt((dataLen / senArrLen)+'') + 1;
    let arrSecLen = dataLen > senArrLen ? senArrLen : dataLen;
    let arrOuter = new Array(arrOuterLen);
    let arrOuterIndex = 0;

    for (let i = 0; i < dataLen; i++) {
        if (i % senArrLen === 0){
            arrOuterIndex++;
            let len = arrSecLen * arrOuterIndex;
            arrOuter[arrOuterIndex-1] = new  Array(dataLen % senArrLen);
            if(arrOuterIndex === arrOuterLen)
                dataLen % senArrLen === 0 ?
                    len = dataLen % senArrLen + senArrLen * arrOuterIndex :
                    len = dataLen % senArrLen + senArrLen * (arrOuterIndex - 1);
            let arrSec_index = 0;
            for (let k = i; k < len; k++) {
                arrOuter[arrOuterIndex-1][arrSec_index] = data[k];
                arrSec_index++;
            }
        }
    }
    return arrOuter;
  };

  getTopClasses(tensor, labels, k = 5, deQuantizeParams) {
    let probs = Array.from(tensor);
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

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

}