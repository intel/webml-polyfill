class baseRunner {
  constructor(modelInfo, processHandle) {
    this.modelInfo = modelInfo
    this.rawModel = null
    this.labels = null
    this.model = null
    this.inputTensor = []
    this.outputTensor = []
    this.processHandle = processHandle
    this.backend = null
    this.prefer = null
    this.initialized = false
    this.loaded = false
    this.resolveGetRequiredOps = null
    this.outstandingRequest = null
    this.requiredOps = null
  }

  init = () => {
    // reset all states
    this.loaded = false
    this.initialized = false
    this.backend = null
    this.prefer = null
    this.initInputOutput()
  }

  initInputOutput = () => {
    let typedArray = this.modelInfo.isQuantized ? Uint8Array : Float32Array
    this.inputTensor = [new typedArray(this.modelInfo.inputSize.reduce((a, b) => a * b))]
    let outputSize = this.modelInfo.outputSize
    if (typeof outputSize === 'number') {
      this.outputTensor = [new typedArray(outputSize)]
    } else {
      this.outputTensor = [new typedArray(outputSize.reduce((a, b) => a * b))]
    }
  }

  loadModel = async () => {
    if (this.loaded) {
      return 'LOADED'
    }
    this.init()
    let result = await loadModelAndLabels(this.modelInfo.modelFile, this.processHandle, this.modelInfo.labelsFile)
    if (result.text != undefined) {
      this.labels = result.text.split('\n')
      console.log(`labels: ${this.labels}`)
    }
    let modelFile = this.modelInfo.modelFile
    switch (modelFile.split('.').pop()) {
      case 'tflite':
        let flatBuffer = new flatbuffers.ByteBuffer(result.bytes)
        this.rawModel = tflite.Model.getRootAsModel(flatBuffer)
        this.rawModel._rawFormat = 'TFLITE'
        printTfLiteModel(this.rawModel)
        break
      case 'onnx':
        let err = onnx.ModelProto.verify(result.bytes)
        if (err) {
          throw new Error(`Invalid model ${err}`)
        }
        this.rawModel = onnx.ModelProto.decode(result.bytes)
        this.rawModel._rawFormat = 'ONNX'
        printOnnxModel(this.rawModel)
        break
      case 'bin':
        const networkFile = modelFile.replace(/bin$/, 'xml')
        const networkText = await loadUrl(networkFile, false, this.processHandle, this.outstandingRequest)
        const weightsBuffer = result.bytes.buffer
        this.rawModel = new OpenVINOModel(networkText, weightsBuffer)
        this.rawModel._rawFormat = 'OPENVINO'
        break
      default:
        throw new Error('Unrecognized model format')
    }
    this.loaded = true
    return 'SUCCESS'
  }

  initModel = async (backend, prefer) => {
    if (!this.loaded) {
      return 'NOT_LOADED'
    }
    if (this.initialized && backend === this.backend && prefer === this.prefer) {
      return 'INITIALIZED'
    }
    this.initialized = false
    this.backend = backend
    this.prefer = prefer
    let postOptions = this.modelInfo.postOptions || {}
    let configs = {
      rawModel: this.rawModel,
      backend: backend,
      prefer: prefer,
      softmax: postOptions.softmax || false,
    }
    switch (this.rawModel._rawFormat) {
      case 'TFLITE':
        this.model = new TFliteModelImporter(configs)
        break
      case 'ONNX':
        this.model = new OnnxModelImporter(configs)
        break
      case 'OPENVINO':
        this.model = new OpenVINOModelImporter(configs)
        break
    }
    let compileStatus = await this.model.createCompiledModel()
    console.log(`compilation status: ${compileStatus}`)
    let start = performance.now()
    let result = await this.model.compute(this.inputTensor, this.outputTensor)
    console.log(`compute result: ${result}`)
    let elapsed = performance.now() - start
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`)
    this.initialized = true
    if (this.resolveGetRequiredOps) {
      this.requiredOps = this.resolveGetRequiredOps(this.model.getRequiredOps())
    }
    return 'SUCCESS'
  }
 
  getRequiredOps = async () => {
    if (!this.initialized) {
      return new Promise(resolve => this.resolveGetRequiredOps = resolve)
    } else {
      return this.model.getRequiredOps()
    }
  }

  getSubgraphsSummary = () => {
    if (this.model._backend !== 'WebML' &&
        this.model._compilation &&
        this.model._compilation._preparedModel) {
      return this.model._compilation._preparedModel.getSubgraphsSummary()
    } else {
      return []
    }
  }

  predict = async (source, options) =>  {
    if (!this.initialized) {
      return
    }
    prepareInputTensor(source, options, this.inputTensor)
    let start = performance.now()
    let result = await this.model.compute(this.inputTensor, this.outputTensor)
    console.log(`Compute result: ${result}`)
    let elapsed = performance.now() - start
    return this.transferResult(elapsed)
  }

  transferResult = (elapsed) => {
    return {
      time: elapsed,
    }
  }

  resetDrawLive = () => {//TODO need optimization
    //
  }

  deleteAll = () => {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll()
    }
  }
}

class semanticSegmentationRunner extends baseRunner {
  constructor(modelInfo, processHandle) {
    super(modelInfo, processHandle)
  }

  initInputOutput = () => {
    let typedArray = this.modelInfo.isQuantized ? Uint8Array : Float32Array
    this.inputTensor = [new typedArray(this.modelInfo.inputSize.reduce((a, b) => a * b))]
    this.outputTensor = [new Int32Array(this.modelInfo.outputSize.reduce((a, b) => a * b))]
  }

  transferResult = (elapsed) => {
    return {
      time: elapsed,
      drawData: {
        data: this.outputTensor[0],
        outputShape: this.modelInfo.outputSize,
        labels: this.labels,
      },
    }
  }
}

export { baseRunner, semanticSegmentationRunner }