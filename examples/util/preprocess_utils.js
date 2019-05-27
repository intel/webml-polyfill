class Preprocessor {
  constructor(source, inputShape, inputType, prefetch = true, preprocessOptions) {

    this.source = source;
    this.late = 0;
    this.ontime = 0;

    this.inputType = inputType || Float32Array;
    this.mean = preprocessOptions.mean || [0, 0, 0];
    this.std  = preprocessOptions.std  || [1, 1, 1];
    this.normFactor = preprocessOptions.norm ? 255 : 1;
    this.channelScheme = preprocessOptions.channelScheme || 'RGB';

    this.width = inputShape[1];
    this.height = inputShape[0];
    this.channels = inputShape[2];
    this.imageChannels = 4; // RGBA

    this.preprocessCanvas = document.createElement('canvas');
    this.preprocessCanvas.width = this.width;
    this.preprocessCanvas.height = this.height;
    this.preprocessContext = this.preprocessCanvas.getContext('2d');

    this.currIndex = 0;
    this.readyIndex = 0;
    this.pendingRequests = [];
    this.bufferQueue = [
      new this.inputType(this.width * this.height * 3),
      new this.inputType(this.width * this.height * 3)
    ];

    if (prefetch) {
      // warm up the queue
      this._getNextFrame();
      this.getFrame = this._getFramePrefetch;
    } else {
      this.getFrame = this._getFrameNoPrefetch;
    }
  }

  async _getFramePrefetch() {
    setTimeout(() => this._getNextFrame(), 0);
    if (++this.currIndex <= this.readyIndex) {
      return this.bufferQueue[this.currIndex % 2];
    }
    return await new Promise(resolve => this.pendingRequests.push(resolve));
  }

  _getFrameNoPrefetch() {
    return this._getNextFrame();
  }

  _getNextFrame() {

    const {
      width,
      height,
      channels,
      mean,
      std,
      normFactor,
      channelScheme,
      imageChannels
    } = this;

    this.preprocessContext.drawImage(this.source, 0, 0, width, height);
    const pixels = this.preprocessContext.getImageData(0, 0, width, height).data;
    const tensor = this.bufferQueue[++this.readyIndex % 2];

    if (channelScheme === 'RGB') {
      // NHWC layout
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            const value = pixels[y*width*imageChannels + x*imageChannels + c] / normFactor;
            tensor[y*width*channels + x*channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else if (channelScheme === 'BGR') {
      // NHWC layout
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            const value = pixels[y*width*imageChannels + x*imageChannels + (channels-c-1)] / normFactor;
            tensor[y*width*channels + x*channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else {
      throw new Error(`Unknown color channel scheme ${channelScheme}`);
    }

    if (this.pendingRequests.length) {
      const resolveFn = this.pendingRequests.shift();
      resolveFn(tensor);
    }

    return tensor;
  }
}






