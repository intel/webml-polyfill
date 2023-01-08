var ie = null;
try {
  ie = require("inference-engine-node")
} catch(e) {
  console.log(e);
}

class OpenVINORunner extends BaseRunner {
  constructor() {
    super();
    this._output = null;
    this._rawModel = null;
    this._ieCore = null;
    this._network = null;
    this._execNet = null;
    this._tensor = null;
    this._postOptions = null;
    this._deQuantizeParams = null;
    this._inputInfo = null;
    this._outputInfo = null;
    this._currentBackend = null;
    if (ie !== null) {
      this._ieCore = ie.createCore();
    }
    // this._configureBackend();
  }

  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  /** @override */
  doInitialization = (modelInfo) => {
    this._setLoadedFlag(false);
    this._setInitializedFlag(false);
    this._setModelInfo(modelInfo);
    this._setDeQuantizeParams([]);
    this._setBackend(null);
  };

  _setDeQuantizeParams = (params) => {
    this._deQuantizeParams = params;
  };

  /** @override */
  _loadModelFile = async (url) => {
    if (this._ieCore !== null) {
      if (url !== undefined) {
        const arrayBuffer = await this._loadURL(url, this._progressHandler, true);
        const bytes = new Uint8Array(arrayBuffer);
        switch (url.split('.').pop()) {
          case 'bin':
            const networkURL = url.replace(/bin$/, 'xml');
            const networkFile = await this._loadURL(networkURL);
            const weightsBuffer = bytes.buffer;
            var network = await this._ieCore.readNetworkFromData(networkFile, weightsBuffer);
            var inputs_info = network.getInputsInfo();
            var outputs_info = network.getOutputsInfo();
            this._network = network;
            this._inputInfo = inputs_info[0];
            let dims = this._inputInfo.getDims();
            if (dims.length === 4) {
              this._inputInfo.setLayout('nhwc');
            }
            this._outputInfo = outputs_info[0];
        }
        this._setLoadedFlag(true);
      } else {
        throw new Error(`There's none model file info, please check config info of modelZoo.`);
      }
    } else {
      throw new Error(`The infernece-engine-node is not worked, please check the Node.js platform is enabled`);
    }
  };

  /** @override */
  _doCompile = async (options) => {
    const modelFormat = this._currentModelInfo.format;
    const device = options.backend;
    if (modelFormat === 'OpenVINO') {
      let exec_net = await this._ieCore.loadNetwork(this._network, device);
      this._execNet = exec_net;
      this._postOptions = this._currentModelInfo.postOptions || {};
    } else {
      throw new Error(`Unsupported '${this._currentModelInfo.format}' input`);
    }
  }

  getDeQuantizeParams = () => {
    return this._deQuantizeParams;
  };

  _getTensor = (input) => {
    const image = input.src;
    const options = input.options;

    image.width = image.videoWidth || image.naturalWidth;
    image.height = image.videoHeight || image.naturalHeight;

    const [height, width, channels] = options.inputSize;
    const preOptions = options.preOptions || {};
    const mean = preOptions.mean || [0, 0, 0, 0];
    const std = preOptions.std || [1, 1, 1, 1];
    const normlizationFlag = preOptions.norm || false;
    const channelScheme = preOptions.channelScheme || 'RGB';
    const imageChannels = options.imageChannels || 4; // RGBA
    const drawOptions = options.drawOptions;

    let canvasElement = document.createElement('canvas');
    canvasElement.width = width;
    canvasElement.height = height;
    let canvasContext = canvasElement.getContext('2d');

    if (drawOptions) {
      canvasContext.drawImage(image, drawOptions.sx, drawOptions.sy, drawOptions.sWidth, drawOptions.sHeight,
        0, 0, drawOptions.dWidth, drawOptions.dHeight);
    } else {
      if (options.scaledFlag) {
        const resizeRatio = Math.max(Math.max(image.width / width, image.height / height), 1);
        const scaledWidth = Math.floor(image.width / resizeRatio);
        const scaledHeight = Math.floor(image.height / resizeRatio);
        canvasContext.drawImage(image, 0, 0, scaledWidth, scaledHeight);
      } else {
        canvasContext.drawImage(image, 0, 0, width, height);
      }
    }

    let pixels = canvasContext.getImageData(0, 0, width, height).data;

    if (normlizationFlag) {
      pixels = new Float32Array(pixels).map(p => p / 255);
    }

    let infer_req = this._execNet.createInferRequest();
    const input_blob = infer_req.getBlob(this._inputInfo.name());
    const input_data = new Float32Array(input_blob.wmap());

    if (channelScheme === 'RGB') {
      if (channels > 1) {
        for (let c = 0; c < channels; ++c) {
          for (let h = 0; h < height; ++h) {
            for (let w = 0; w < width; ++w) {
              let value = pixels[h * width * imageChannels + w * imageChannels + c];
              input_data[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
            }
          }
        }
      } else if (channels === 1) {
        for (let c = 0; c < channels; ++c) {
          for (let h = 0; h < height; ++h) {
            for (let w = 0; w < width; ++w) {
              let index = h * width * imageChannels + w * imageChannels + c;
              let value = (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3;
              input_data[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
            }
          }
        }
      }
    } else if (channelScheme === 'BGR') {
      for (let c = 0; c < channels; ++c) {
        for (let h = 0; h < height; ++h) {
          for (let w = 0; w < width; ++w) {
            let value = pixels[h * width * imageChannels + w * imageChannels + (channels - c - 1)];
            input_data[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else {
      throw new Error(`Unsupport '${channelScheme}' Color Channel Scheme `);
    }

    input_blob.unmap();
    this._inferReq = infer_req;
  }

  /**
   * This method is to get downsample audio buffer.
   */
  _downsampleAudioBuffer = (buffer, rate, baseRate) => {
    if (rate == baseRate) {
      return buffer;
    }

    if (baseRate > rate) {
      throw "downsampling rate show be smaller than original sample rate";
    }

    const sampleRateRatio = Math.round(rate / baseRate);
    const newLength = Math.round(buffer.length / sampleRateRatio);
    let abuffer = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < abuffer.length) {
      let nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      abuffer[offsetResult] = accum / count;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return abuffer;
  };

  /**
   * This method is to get audio mfccs array.
   */
  _getAudioMfccs = (pcm, sampleRate, windowSize, windowStride,
                    upperFrequencyLimit = 4000,
                    lowerFrequencyLimit = 20,
                    filterbankChannelCount = 40,
                    dctCoefficientCount = 13) => {
    let pcmPtr = Module._malloc(8 * pcm.length);
    let lenPtr = Module._malloc(4);

    for (let i = 0; i < pcm.length; i++) {
      Module.HEAPF64[pcmPtr / 8 + i] = pcm[i];
    };

    Module.HEAP32[lenPtr / 4] = pcm.length;
    let tfMfccs = Module.cwrap('tf_mfccs', 'number',
          ['number', 'number', 'number', 'number',
          'number', 'number', 'number', 'number', 'number']);
    let mfccsPtr = tfMfccs(pcmPtr, lenPtr, sampleRate, windowSize,
          windowStride, upperFrequencyLimit, lowerFrequencyLimit,
          filterbankChannelCount, dctCoefficientCount);
    let mfccsLen = Module.HEAP32[lenPtr >> 2];
    let audioMfccs = [mfccsLen];

    for (let i = 0; i < mfccsLen; i++) {
      audioMfccs[i] = Module.HEAPF64[(mfccsPtr >> 3) + i];
    }

    Module._free(pcmPtr, lenPtr, mfccsPtr);
    return audioMfccs;
  };

  _getTensorByAudio = async (input) => {
    const audio = input.src;
    const options = input.options;
    const sampleRate = options.sampleRate;
    const mfccsOptions = options.mfccsOptions;
    const inputSize = options.inputSize.reduce((a, b) => a * b);

    let audioContext = new (window.AudioContext || window.webkitAudioContext)();
    let rate = audioContext.sampleRate;

    let request = new Request(audio.src);
    let response = await fetch(request);
    let audioFileData = await response.arrayBuffer();
    let audioDecodeData = await audioContext.decodeAudioData(audioFileData);
    let audioPCMData = audioDecodeData.getChannelData(0);
    let abuffer = this._downsampleAudioBuffer(audioPCMData, rate, sampleRate);

    if (typeof mfccsOptions !== 'undefined') {
      abuffer = this._getAudioMfccs(abuffer,
                                    sampleRate,
                                    mfccsOptions.windowSize,
                                    mfccsOptions.windowStride,
                                    mfccsOptions.upperFrequencyLimit,
                                    mfccsOptions.lowerFrequencyLimit,
                                    mfccsOptions.filterbankChannelCount,
                                    mfccsOptions.dctCoefficientCount);
    }

    let infer_req = this._execNet.createInferRequest();
    const input_blob = infer_req.getBlob(this._inputInfo.name());
    const input_data = new Float32Array(input_blob.wmap());

    let inputDataLen = input_data.length;
    let abufferLen = abuffer.length;
    const maxLen = inputDataLen > abufferLen? inputDataLen : abufferLen;

    for (let i = 0; i < maxLen; i++) {
      if (i > inputDataLen) {
        break;
      } else if (i >= abufferLen) {
        input_data[i] = 0;
      } else {
        input_data[i] = abuffer[i];
      }
    }

    input_blob.unmap();
    this._inferReq = infer_req;
  }

  /** @override */
  _getInputTensor = async (input) => {
    if (input.src.tagName === 'AUDIO') {
      await this._getTensorByAudio(input);
    } else {
      this._getTensor(input);
    }
  };

  _getOutputTensorTypedArray = () => {
    const typedArray = this._currentModelInfo.isQuantized || false ? (this._currentModelInfo.isDNNL || this._currentModelInfo.isIE || false ? Float32Array : Uint8Array) : Float32Array;
    return typedArray;
  };

  /** @override */
  _getOutputTensor = () => {
    return this._output;
  };

  /** @override */
  _doInference = async () => {
    await this._inferReq.startAsync();
    const output_blob = this._inferReq.getBlob(this._outputInfo.name());
    const typedArray = this._getOutputTensorTypedArray();
    const output_data = new typedArray(output_blob.rmap());
    this._output = output_data;
  };
}