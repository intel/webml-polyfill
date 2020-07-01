class SkeletonDetectionWebNNExecutor extends WebNNExecutor {
  constructor() {
    super();
    this._modelConfig = null;
  }

  _setModelConfig = (config) => {
    this._modelConfig = config;
  };

  /** @override */
  _createRunner = () => {
    const runner = new SkeletonDetectionRunner();
    return runner;
  };

  /** @override */
  loadAndCompileModel = async (modelId, coModelId) => {
    if (this._modelConfig === null) {
      let posenetConfigURL = './resources/posenetConfig.json';
      let request = new Request(posenetConfigURL);
      let response = await fetch(request);
      let configDic = await response.json();
      this._setModelConfig(configDic);
      this._setModelInfo(humanPoseEstimationModels[0]);
    }

    await this._runner.loadAndCompileModel(this._currentBackend.replace('WebNN', 'WebML'),
                                           this._currentPrefer,
                                           this._currentModelInfo,
                                           this._modelConfig,
                                           this._bEagerMode,
                                           this._supportedOps,
                                           true);
  };

  /** @override */
  _executeSingle = async () => {
    const inputSize = this._currentModelInfo.inputSize;
    const outputStride = Number(this._modelConfig.outputStride);
    const scaleFactor = this._modelConfig.scaleFactor;
    const scaleWidth = getValidResolution(scaleFactor, inputSize[1], outputStride);
    const scaleHeight = getValidResolution(scaleFactor, inputSize[0], outputStride);
    const input = {
      src: this._currentInputElement,
      options: {
        inputSize: [scaleHeight, scaleWidth, inputSize[2]],
        preOptions: this._currentModelInfo.preOptions,
        imageChannels: 4,
      },
    };
    await this._runner.run(input);
  };

  /** @override */
  _postProcess = (data) => {
    const drawPoses = (src, canvas, poses, options) => {
      const ctx = canvas.getContext('2d');
      const width = src.naturalWidth;
      const height = src.naturalHeight;
      canvas.setAttribute('width', width);
      canvas.setAttribute('height', height);
      ctx.drawImage(src, 0, 0, width, height);
      if (poses.length == 0) {
        return;
      }
      poses.forEach((pose) => {
        const scaleX = canvas.width / options.scaleWidth;
        const scaleY = canvas.height / options.scaleHeight;
        if (pose.score >= options.minScore) {
          if (options.showPose) {
            drawKeypoints(pose.keypoints, options.minScore, ctx, scaleX, scaleY);
            drawSkeleton(pose.keypoints, options.minScore, ctx, scaleX, scaleY);
          }
          if (options.showBoundingBox) {
            drawBoundingBox(pose.keypoints, ctx, scaleX, scaleY);
          }
        }
      });
    };

    const output = this._runner.getOutput();
    const inputSize = this._currentModelInfo.inputSize;
    const outputStride = Number(this._modelConfig.outputStride);
    const scaleFactor = this._modelConfig.scaleFactor;
    const scaleWidth = getValidResolution(scaleFactor, inputSize[1], outputStride);
    const scaleHeight = getValidResolution(scaleFactor, inputSize[0], outputStride);
    const scaleInputSize = [1, scaleHeight, scaleWidth, this._currentModelInfo.inputSize[2]];
    const start = performance.now();
    const singlePose = decodeSinglepose(sigmoid(output.heatmapTensor),
                                        output.offsetTensor,
                                        toHeatmapsize(scaleInputSize, outputStride),
                                        outputStride);
    this._setDecodeTime(performance.now() - start);
    let options = {
      inputHeight: this._currentModelInfo.inputSize[0],
      inputWidth: this._currentModelInfo.inputSize[1],
      scaleHeight: scaleHeight,
      scaleWidth: scaleWidth,
      minScore: this._modelConfig.minScore,
      showPose: true,
      showBoundingBox: false,
    }
    const canvasSingle = document.getElementById('showcanvas');
    drawPoses(this._currentInputElement, canvasSingle, singlePose, options);
  };
}
