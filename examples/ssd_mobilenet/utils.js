const INPUT_TENSOR_SIZE = 300*300*3;
const NUM_BOXES = 1083 + 600 + 150 + 54 + 24 + 6;
const BOX_SIZE = 4;
const NUM_CLASSES = 91;
const OUTPUT_TENSOR_SIZE = NUM_BOXES * (BOX_SIZE + NUM_CLASSES);
const MODEL_FILE = './model/ssd_mobilenet.tflite';
const LABELS_FILE = './model/coco_labels_list.txt';

class Utils {
  constructor() {
    this.tfModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputBoxTensor;
    this.outputClassScoresTensor;
    this.anchors;

    this.inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    this.outputBoxTensor = new Float32Array(NUM_BOXES * BOX_SIZE);
    this.outputClassScoresTensor = new Float32Array(NUM_BOXES * NUM_CLASSES);
    this.container = document.getElementById('container');
    this.progressBar = document.getElementById('progressBar');
    this.progressContainer = document.getElementById('progressContainer');
    this.canvasElement = document.getElementById('canvas');
    this.canvasContext = this.canvasElement.getContext('2d');
    this.canvasShowElement = document.getElementById('canvasShow');

    this.initialized = false;
  }

  async init(backend) {
    this.initialized = false;
    let result;
    this._generateAnchors({});
    if (!this.tfModel) {
      result = await this.loadModelAndLabels(MODEL_FILE, LABELS_FILE);
      this.container.removeChild(progressContainer);
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
      // printTfLiteModel(this.tfModel);
    }
    this.model = new SsdMobileNet(this.tfModel, backend);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute(this.inputTensor, this.outputBoxTensor, this.outputClassScoresTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict(imageSource) {
    if (!this.initialized) return;
    this.canvasContext.drawImage(imageSource, 0, 0,
                                 this.canvasElement.width,
                                 this.canvasElement.height);
    // console.log('inputTensor1', this.inputTensor)
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    // console.log('inputTensor2', this.inputTensor)
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputBoxTensor, this.outputClassScoresTensor);
    // console.log('outputBoxTensor', this.outputBoxTensor)
    // console.log('outputClassScoresTensor', this.outputClassScoresTensor)
    // let startDecode = performance.now();
    this._decodeOutputBoxTensor();
    // console.log(`Decode time: ${(performance.now() - startDecode).toFixed(2)} ms`);
    // let startNMS = performance.now();
    let [boxesList, scoresList, classesList] = this._NMS({});
    // console.log(`NMS time: ${(performance.now() - startNMS).toFixed(2)} ms`);
    // let startVisual = performance.now();
    this._visualize(imageSource, boxesList, scoresList, classesList);
    // console.log(`visual time: ${(performance.now() - startVisual).toFixed(2)} ms`);
    let elapsed = performance.now() - start;
    console.log(`Inference time: ${elapsed.toFixed(2)} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: ${elapsed.toFixed(2)} ms`;
  }

  async loadModelAndLabels(modelUrl, labelsUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(labelsUrl);
    return {bytes: bytes, text: text};
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      let request = new XMLHttpRequest();
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        if (request.readyState === 4) {
          if (request.status === 200) {
              resolve(request.response);
          } else {
              reject(new Error('Failed to load ' + modelUrl + ' status: ' + request.status));
          }
        }
      };
      if (progress) {
        let self = this;
        request.onprogress = function(ev) {
          if (ev.lengthComputable) {
            let percentComplete = ev.loaded / ev.total * 100;
            percentComplete = percentComplete.toFixed(0);
            self.progressBar.style = `width: ${percentComplete}%`;
            self.progressBar.innerHTML = `${percentComplete}%`;
          }
        }
      }
      request.send();
    });
  }

  prepareInputTensor(tensor, canvas) {
    const width = 300;
    const height = 300;
    const channels = 3;
    const imageChannels = 4; // RGBA
    const mean = 127.5;
    const std = 127.5;
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) or canvas.height(${canvas.height}) is not 300`);
    }
    let context = canvas.getContext('2d');
    let pixels = context.getImageData(0, 0, width, height).data;
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = (value - mean)/std;
        }
      }
    }
  }

  /**
  * Decode out box coordinate
  * See tensorflow ssd_mobilenet_v1 example for details:
  * https://github.com/tensorflow/models/blob/master/research/object_detection/box_coders/faster_rcnn_box_coder.py
  * 
  */
  _decodeOutputBoxTensor() {
    if (this.outputBoxTensor.length % BOX_SIZE !== 0) {
      throw new Error(`The length 0f outputTensorDecode should be the multiple of ${BOX_SIZE}!`);
    }
    
    // scale_factors: [y_scale, x_scale, height_scale, width_scale]
    const scale_factors = [10.0, 10.0, 5.0, 5.0];
    let boxOffset = 0;
    let ty, tx, th, tw, w, h, ycenter, xcenter;
    for (let y = 0; y < NUM_BOXES; ++y) {
      const [ycenter_a, xcenter_a, ha, wa] = this.anchors[y]
      ty = this.outputBoxTensor[boxOffset] / scale_factors[0];
      tx = this.outputBoxTensor[boxOffset + 1] / scale_factors[1];
      th = this.outputBoxTensor[boxOffset + 2] / scale_factors[2];
      tw = this.outputBoxTensor[boxOffset + 3] / scale_factors[3];
      w = Math.exp(tw) * wa;
      h = Math.exp(th) * ha;
      ycenter = ty * ha + ycenter_a;
      xcenter = tx * wa + xcenter_a;
      // Decoded box coordinate: [ymin, xmin, ymax, xmax]
      this.outputBoxTensor[boxOffset] = ycenter - h / 2;
      this.outputBoxTensor[boxOffset + 1] = xcenter - w / 2;
      this.outputBoxTensor[boxOffset + 2] = ycenter + h / 2;
      this.outputBoxTensor[boxOffset + 3] = xcenter + w / 2;
      boxOffset += BOX_SIZE;
    }
  }

  /**
  * Get IOU(intersection-over-union) of 2 boxes
  *
  * @param {number[4]} boxCord1 - An 4 element Array of box coordinate.
  * @param {number[4]} boxCord2 - An 4 element Array of box coordinate.
  * @returns {number} IOU
  */
  _IOU(boxCord1, boxCord2) {
    if (boxCord1.length !== 4 || boxCord2.length !== 4) {
      throw new Error('[box_decode] Each input length should be 4!');
    }

    const [ymin1, xmin1, ymax1, xmax1] = boxCord1;
    const [ymin2, xmin2, ymax2, xmax2] = boxCord2;
    let minYmax = Math.min(ymax1, ymax2);
    let maxYmin = Math.max(ymin1, ymin2);
    let height = Math.max(0, minYmax - maxYmin);
    let minXmax = Math.min(xmax1, xmax2);
    let maxXmin = Math.max(xmin1, xmin2);
    let width = Math.max(0, minXmax - maxXmin);
    let intersection = height * width;
    let area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    let area2 = (ymax2 - ymin2) * (xmax2 - xmin2);
    let areaSum = area1 + area2 - intersection
    if (areaSum === 0) {
      throw new Error('[IOU] areaSum can not be 0!');
    }
    let IOU = intersection / areaSum;
    return IOU;
  }

  /**
  * Generate anchors
  * See tensorflow ssd_mobilenet_v1 example for details:
  * https://github.com/tensorflow/models/blob/master/research/object_detection/anchor_generators/multiple_grid_anchor_generator.py
  * https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config
  * 
  */
  _generateAnchors(options) {
  const {
    min_scale = 0.2,
    max_scale = 0.95,
    aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333],
    base_anchor_size = [1.0, 1.0],
    feature_map_shape_list = [[19, 19], [10, 10], [5, 5], [3, 3], [2, 2], [1, 1]],
    interpolated_scale_aspect_ratio = 1.0,
    reduce_boxes_in_lowest_layer=true
  } = options;
  const num_layers = feature_map_shape_list.length;
  let box_specs_list = [];

  let scales = [];

  for (let i = 0; i < num_layers; ++i) {
    let scale = min_scale + (max_scale - min_scale) * i / (num_layers - 1);
    scales.push(scale);
  }
  // console.log(scales)

  scales.forEach((scale, i) => {
    let scale_next = (i === scales.length - 1) ? 1.0 : scales[i + 1];
    let layer_box_specs = [];
    if (i === 0 && reduce_boxes_in_lowest_layer) {
      layer_box_specs = [[0.1, 1.0], [scale, 2.0], [scale, 0.5]]
    } else {
      aspect_ratios.forEach((aspect_ratio, j) => {
        layer_box_specs.push([scale, aspect_ratio])
      });
      if (interpolated_scale_aspect_ratio > 0.0) {
        layer_box_specs.push([Math.sqrt(scale * scale_next),
                                interpolated_scale_aspect_ratio])
      }
    }
    box_specs_list.push(layer_box_specs);
  });

  let anchors = [];
  for (let i = 0; i < num_layers; ++i) {
    const grid_height = feature_map_shape_list[i][0];
    const grid_width = feature_map_shape_list[i][1];
    let anchor_stride = [1.0 / grid_height, 1.0 / grid_width];
    let anchor_offset = [anchor_stride[0] / 2, anchor_stride[1] / 2];

    for (let h = 0; h < grid_height; ++h) {
      for (let w = 0; w < grid_width; ++w) {
        box_specs_list[i].forEach((layer_box_spec, j) => {
          const [scale, aspect_ratio] = layer_box_spec;
          let ratio_sqrt = Math.sqrt(aspect_ratio);
          let y_center = h * anchor_stride[0] + anchor_offset[0];
          let x_center = w * anchor_stride[1] + anchor_offset[1];
          let height = scale / ratio_sqrt * base_anchor_size[0];
          let width = scale * ratio_sqrt * base_anchor_size[0];
          anchors.push([y_center, x_center, height, width]);
        });
      }
    }
  }
  // console.log('box_specs_list', box_specs_list)
  // console.log('anchors', anchors)
  this.anchors = anchors;
  }

  /**
  * NMS(Non Max Suppression)
  * See tensorflow ssd_mobilenet_v1 example for details:
  * https://github.com/tensorflow/models/blob/master/research/object_detection/core/post_processing.py#L38
  *
  * @param {object} options - Some options.
  */
  _NMS(options) {
    // Using a little higher threshold and lower max detections can save inference time with little performance loss.
    const {
      score_threshold = 0.1, // 1e-8
      iou_threshold = 0.6,
      max_detections_per_class = 10, // 100
      max_total_detections = 100
    } = options;

    let boxesList = [];
    let scoresList = [];
    let classesList = [];
    
    // Skip background 0
    for (let x = 1; x < NUM_CLASSES; ++x) {
      // let startNMS = performance.now();
      let boxes = [];
      let scores = [];
      for (let y = 0; y < NUM_BOXES; ++y) {
        let scoreIndex = y * NUM_CLASSES + x;
        if (this.outputClassScoresTensor[scoreIndex] > score_threshold) {
          let boxIndexStart = y * BOX_SIZE;
          boxes.push(this.outputBoxTensor.subarray(boxIndexStart, boxIndexStart + BOX_SIZE));
          scores.push(this.outputClassScoresTensor[scoreIndex]);
        }
      }
      // console.log(`NMS time${x}: ${(performance.now() - startNMS).toFixed(2)} ms`);
      let boxForClassi = [];
      let scoreForClassi = [];
      let classi = [];
      // console.log('boxes', boxes);
      // console.log('scores', scores);
      while (scores.length !== 0 && scoreForClassi.length < max_detections_per_class) {
        let max = 0;
        let maxIndex = 0;
        // Find max score
        scores.forEach((score, j) => {
          if (score > max) {
            max = score;
            maxIndex = j;
          }
        });
        // Push and delete max
        let maxBox = boxes[maxIndex];
        boxForClassi.push(boxes.splice(maxIndex, 1)[0]);
        scoreForClassi.push(scores.splice(maxIndex, 1)[0]);
        classi.push(x);
        let retainBoxes = []; 
        let retainScores = [];
        boxes.forEach((box, j) => {
          if (this._IOU(box, maxBox) < iou_threshold) {
            // Remain low IOU and delete high IOU
            retainBoxes.push(boxes[j]);
            retainScores.push(scores[j]);
          }
        });
        boxes = retainBoxes;
        scores = retainScores;
      }
      // console.log('boxForClassi', boxForClassi);
      // console.log('scoreForClassi', scoreForClassi);
      // console.log('classi', classi);
      boxesList = boxesList.concat(boxForClassi);
      scoresList = scoresList.concat(scoreForClassi);
      classesList = classesList.concat(classi);
      // console.log(`boxesList`, boxesList)
      // console.log(`scoresList`, scoresList)
      // console.log(`classesList`, classesList)
      // console.log(`NMS time${x}: ${(performance.now() - startNMS).toFixed(2)} ms`);
    }

    if (scoresList.length > max_total_detections) {
      this.totalDetections = max_total_detections;
      // quickSort get max_total detections
      let low = 0;
      let high = scoresList.length - 1;
      while (low < high) {
        let i = low;
        let j = high;
        let tmpScore = scoresList[i];
        let tmpBox = boxesList[i];
        let tmpClassi = classesList[i];
        while (i < j) {
          while (i < j && scoresList[j] < tmpScore) {
            --j;
          }
          if (i < j) {
            scoresList[i] = scoresList[j];
            boxesList[i] = boxesList[j];
            classesList[i] = classesList[j];
            ++i;
          }
          while (i < j && scoresList[i] > tmpScore) {
            ++i;
          }
          if (i < j) {
            scoresList[j] = scoresList[i];
            boxesList[j] = boxesList[i];
            classesList[j] = classesList[i];
            --j;
          }
        }
        scoresList[i] = tmpScore;
        boxesList[i] = tmpBox;
        classesList[i] = tmpClassi;
        if (i === max_total_detections) {
          low = high;
        } else if (i < max_total_detections) {
          low = i + 1;
        } else {
          high = i - 1;
        }
      }
    } else {
      this.totalDetections = scoresList.length;
    }
    // console.log(`boxesList`, boxesList)
    // console.log(`scoresList`, scoresList)
    // console.log(`classesList`, classesList)
    return [boxesList, scoresList, classesList];
  }

  /**
  * Draw img and box
  *
  * @param {object} imageSource - Input image element
  */
  _visualize(imageSource, boxesList, scoresList, classesList) {
    let ctx = this.canvasShowElement.getContext('2d');
    if (imageSource.width) {
      this.canvasShowElement.width = imageSource.width / imageSource.height * this.canvasShowElement.height;
    } else {
      this.canvasShowElement.width = imageSource.videoWidth / imageSource.videoHeight * this.canvasShowElement.height;
    }

    let colors = ['red', 'blue', 'green', 'yellowgreen', 'purple', 'orange'];
    ctx.drawImage(imageSource, 0, 0, 
                      this.canvasShowElement.width,
                      this.canvasShowElement.height);
    for (let i = 0; i < this.totalDetections; ++i) {
      // Skip background and blank
      let label = this.labels[classesList[i]];
      if (label !== '???') {
        let [ymin, xmin, ymax, xmax] = boxesList[i];
        ymin = Math.max(0, ymin);
        xmin = Math.max(0, xmin);
        ymax = Math.min(1, ymax);
        xmax = Math.min(1, xmax);
        ymin *= this.canvasShowElement.height;
        xmin *= this.canvasShowElement.width;
        ymax *= this.canvasShowElement.height;
        xmax *= this.canvasShowElement.width;
        let prob = 1 / (1 + Math.exp(-scoresList[i]));

        ctx.strokeStyle = colors[classesList[i] % colors.length];
        ctx.fillStyle = colors[classesList[i] % colors.length];
        ctx.lineWidth = 3;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
        ctx.font = "20px Arial";
        let text = `${label}: ${prob.toFixed(2)}`;
        let width = ctx.measureText(text).width;
        ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10), width + 4, parseInt(ctx.font, 10));
        ctx.fillStyle = "white";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin, ymin - 3);
      }
    }
  }
}