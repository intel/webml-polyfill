class faceDetectorRunner extends baseRunner {
  constructor(modelInfo, processHandle) {
    super(modelInfo, processHandle);
    this.anchors;
    this.outputBoxTensor;
    this.outputClassScoresTensor;
  }

  initInputOutput = () => {
    let inputSize = this.modelInfo.inputSize;
    let numClasses = this.modelInfo.num_classes;
    if (this.modelInfo.type === 'SSD') {
      this.anchors = generateAnchors({});
      let boxSize = this.modelInfo.box_size;
      let numBoxes = this.modelInfo.num_boxes;
      let typedArray;
      if (this.modelInfo.isQuantized) {
        typedArray = Uint8Array;
      } else {
        typedArray = Float32Array;
      }
      this.inputTensor = [new typedArray(inputSize.reduce((a, b) => a * b))];
      this.outputBoxTensor = new typedArray(numBoxes * boxSize);
      this.outputClassScoresTensor = new typedArray(numBoxes * numClasses);
      this.outputTensor = this.prepareSsdOutputTensor();
    } else {
      this.anchors = this.modelInfo.anchors;
      this.inputTensor = [new Float32Array(inputSize.reduce((a, b) => a * b))];
      this.outputTensor = [new Float32Array(this.modelInfo.outputSize)];
    }
  }

  prepareSsdOutputTensor = () => {
    let outputTensor = [];
    const outH = [1083, 600, 150, 54, 24, 6];
    const boxLen = 4;
    const classLen = 2;
    let boxOffset = 0;
    let classOffset = 0;
    let boxTensor;
    let classTensor;
    for (let i = 0; i < 6; ++i) {
      boxTensor = this.outputBoxTensor.subarray(boxOffset, boxOffset + boxLen * outH[i]);
      classTensor = this.outputClassScoresTensor.subarray(classOffset, classOffset + classLen * outH[i]);
      outputTensor[2 * i] = boxTensor;
      outputTensor[2 * i + 1] = classTensor;
      boxOffset += boxLen * outH[i];
      classOffset += classLen * outH[i];
    }
    return outputTensor;
  }

  getFaceBoxes = (source) => {
    if (this.modelInfo.type === 'SSD') {
      decodeOutputBoxTensor({}, this.outputBoxTensor, this.anchors);
      let [totalDetections, boxesList, scoresList, classesList] = NMS({num_classes: 2}, this.outputBoxTensor, this.outputClassScoresTensor);
      boxesList = cropSSDBox(source, totalDetections, boxesList, this.modelInfo.margin);
      let outputBoxes = [];
      for (let i = 0; i < totalDetections; ++i) {
        let [ymin, xmin, ymax, xmax] = boxesList[i];
        ymin = Math.max(0, ymin) * source.height;
        xmin = Math.max(0, xmin) * source.width;
        ymax = Math.min(1, ymax) * source.height;
        xmax = Math.min(1, xmax) * source.width;
        let prob = 1 / (1 + Math.exp(-scoresList[i]));
        outputBoxes.push([xmin, xmax, ymin, ymax, prob]);
      }
      return outputBoxes;
    }
    else {
      let decode_out = decodeYOLOv2({nb_class: 1}, this.outputTensor[0], this.anchors);
      let outputBoxes = getBoxes(decode_out, this.modelInfo.margin);
      for (let i = 0; i < outputBoxes.length; ++i) {
        let [xmin, xmax, ymin, ymax, prob] = outputBoxes[i].slice(1, 6);
        xmin = Math.max(0, xmin) * source.width;
        xmax = Math.min(1, xmax) * source.width;
        ymin = Math.max(0, ymin) * source.height;
        ymax = Math.min(1, ymax) * source.height;
        outputBoxes[i] = [xmin, xmax, ymin, ymax, prob];
      }
      return outputBoxes;
    }
  }
}