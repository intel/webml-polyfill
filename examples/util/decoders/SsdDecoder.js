/**
* Decode out box coordinate
* See tensorflow ssd_mobilenet_v1 example for details:
* https://github.com/tensorflow/models/blob/master/research/object_detection/box_coders/faster_rcnn_box_coder.py
*
*/
function decodeOutputBoxTensor(options, outputBoxTensor, anchors) {
  const {
    box_size = 4,
    num_boxes = 1083 + 600 + 150 + 54 + 24 + 6
  } = options;

  if (outputBoxTensor.length % box_size !== 0) {
    throw new Error(`The length 0f outputTensorDecode should be the multiple of ${box_size}!`);
  }

  // scale_factors: [y_scale, x_scale, height_scale, width_scale]
  const scale_factors = [10.0, 10.0, 5.0, 5.0];
  let boxOffset = 0;
  let ty, tx, th, tw, w, h, ycenter, xcenter;
  for (let y = 0; y < num_boxes; ++y) {
    const [ycenter_a, xcenter_a, ha, wa] = anchors[y]
    ty = outputBoxTensor[boxOffset] / scale_factors[0];
    tx = outputBoxTensor[boxOffset + 1] / scale_factors[1];
    th = outputBoxTensor[boxOffset + 2] / scale_factors[2];
    tw = outputBoxTensor[boxOffset + 3] / scale_factors[3];
    w = Math.exp(tw) * wa;
    h = Math.exp(th) * ha;
    ycenter = ty * ha + ycenter_a;
    xcenter = tx * wa + xcenter_a;
    // Decoded box coordinate: [ymin, xmin, ymax, xmax]
    outputBoxTensor[boxOffset] = ycenter - h / 2;
    outputBoxTensor[boxOffset + 1] = xcenter - w / 2;
    outputBoxTensor[boxOffset + 2] = ycenter + h / 2;
    outputBoxTensor[boxOffset + 3] = xcenter + w / 2;
    boxOffset += box_size;
  }
}

/**
* Get IOU(intersection-over-union) of 2 boxes
*
* @param {number[4]} boxCord1 - An 4 element Array of box coordinate.
* @param {number[4]} boxCord2 - An 4 element Array of box coordinate.
* @returns {number} IOU
*/
function IOU(boxCord1, boxCord2) {
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
function generateAnchors(options) {
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
  return anchors;
}

/**
* NMS(Non Max Suppression)
* See tensorflow ssd_mobilenet_v1 example for details:
* https://github.com/tensorflow/models/blob/master/research/object_detection/core/post_processing.py#L38
*
* @param {object} options - Some options.
*/
function NMS(options, outputBoxTensor, outputClassScoresTensor) {
  // Using a little higher threshold and lower max detections can save inference time with little performance loss.
  const {
    score_threshold = 0.1, // 1e-8
    iou_threshold = 0.5,
    max_detections_per_class = 10, // 100
    max_total_detections = 100,
    num_boxes = 1083 + 600 + 150 + 54 + 24 + 6,
    num_classes = 91,
    box_size = 4
  } = options;

  let totalDetections = null;
  let boxesList = [];
  let scoresList = [];
  let classesList = [];

  // Skip background 0
  for (let x = 1; x < num_classes; ++x) {
    // let startNMS = performance.now();
    let boxes = [];
    let scores = [];
    for (let y = 0; y < num_boxes; ++y) {
      let scoreIndex = y * num_classes + x;
      if (outputClassScoresTensor[scoreIndex] > score_threshold) {
        let boxIndexStart = y * box_size;
        boxes.push(outputBoxTensor.subarray(boxIndexStart, boxIndexStart + box_size));
        scores.push(outputClassScoresTensor[scoreIndex]);
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
        if (IOU(box, maxBox) < iou_threshold) {
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
    totalDetections = max_total_detections;
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
    totalDetections = scoresList.length;
  }
  // console.log(`boxesList`, boxesList)
  // console.log(`scoresList`, scoresList)
  // console.log(`classesList`, classesList)
  return [totalDetections, boxesList, scoresList, classesList];
}

/**
* Draw img and box
*
* @param {object} imageSource - Input image element
*/
function visualize(canvasShowElement, totalDetections, imageSource, boxesList, scoresList, classesList, labels) {
  let ctx = canvasShowElement.getContext('2d');
  if (imageSource.width) {
    canvasShowElement.width = imageSource.width / imageSource.height * canvasShowElement.height;
  } else {
    canvasShowElement.width = imageSource.videoWidth / imageSource.videoHeight * canvasShowElement.height;
  }

  let colors = ['red', 'blue', 'green', 'yellowgreen', 'purple', 'orange'];
  ctx.drawImage(imageSource, 0, 0,
                canvasShowElement.width,
                canvasShowElement.height);
  for (let i = 0; i < totalDetections; ++i) {
    // Skip background and blank
    let label = labels[classesList[i]];
    if (label !== '???') {
      let [ymin, xmin, ymax, xmax] = boxesList[i];
      ymin = Math.max(0, ymin);
      xmin = Math.max(0, xmin);
      ymax = Math.min(1, ymax);
      xmax = Math.min(1, xmax);
      ymin *= canvasShowElement.height;
      xmin *= canvasShowElement.width;
      ymax *= canvasShowElement.height;
      xmax *= canvasShowElement.width;
      let prob = 1 / (1 + Math.exp(-scoresList[i]));

      ctx.strokeStyle = colors[classesList[i] % colors.length];
      ctx.fillStyle = colors[classesList[i] % colors.length];
      ctx.lineWidth = 3;
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
      ctx.font = "20px Arial";
      let text = `${label}: ${prob.toFixed(2)}`;
      let width = ctx.measureText(text).width;
      if (xmin >= 2 && ymin >= parseInt(ctx.font, 10)) {
        ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10), width + 4, parseInt(ctx.font, 10));
        ctx.fillStyle = "white";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin, ymin - 3);
      } else {
        ctx.fillRect(xmin + 2, ymin , width + 4,  parseInt(ctx.font, 10));
        ctx.fillStyle = "white";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin + 2, ymin + 15);
      }
    }
  }
}
