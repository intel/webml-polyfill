const INPUT_TENSOR_SIZE = 300*300*3;
const OUTPUT_TENSOR_SIZE = (1083 + 600 + 150 + 54 + 24 + 6) * (4 + 91);
const MODEL_FILE = './model/ssd_mobilenet.tflite';
const LABELS_FILE = './model/coco_labels_list.txt';

class Utils {
  constructor() {
    this.tfModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;

    this.inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    this.outputTensor = new Float32Array(OUTPUT_TENSOR_SIZE);
    this.container = document.getElementById('container');
    this.progressBar = document.getElementById('progressBar');
    this.progressContainer = document.getElementById('progressContainer');
    this.canvasElement = document.getElementById('canvas');
    this.canvasContext = this.canvasElement.getContext('2d');

    this.initialized = false;
  }

  async init(backend) {
    this.initialized = false;
    let result;
    if (!this.tfModel) {
      result = await this.loadModelAndLabels(MODEL_FILE, LABELS_FILE);
      this.container.removeChild(progressContainer);
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
      printTfLiteModel(this.tfModel);
    }
    this.model = new MobileNet(this.tfModel, backend);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict(imageSource) {
    if (!this.initialized) return;
    this.canvasContext.drawImage(imageSource, 0, 0,
                                 this.canvasElement.width,
                                 this.canvasElement.height);
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    console.log(this.outputTensor)
    let elapsed = performance.now() - start;
    // let classes = this.getTopClasses(this.outputTensor, this.labels, 3);
    console.log(`Inference time: ${elapsed.toFixed(2)} ms`);
    // let inferenceTimeElement = document.getElementById('inferenceTime');
    // inferenceTimeElement.innerHTML = `inference time: ${elapsed.toFixed(2)} ms`;
    // console.log(`Classes: `);
    // classes.forEach((c, i) => {
    //   console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
    //   let labelElement = document.getElementById(`label${i}`);
    //   let probElement = document.getElementById(`prob${i}`);
    //   labelElement.innerHTML = `${c.label}`;
    //   probElement.innerHTML = `${c.prob}%`;
    // });
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
    const width = 224;
    const height = 224;
    const channels = 3;
    const imageChannels = 4; // RGBA
    const mean = 127.5;
    const std = 127.5;
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) or canvas.height(${canvas.height}) is not 224`);
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
  
  getTopClasses(tensor, labels, k = 5) {
    let probs = Array.from(tensor);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
      if (a[0] === b[0]) {return 0;}
      return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    let classes = [];
    for (let i = 0; i < k; ++i) {
      let prob = sorted[i][0];
      let index = sorted[i][1];
      let c = {
        label: labels[index],
        prob: (prob * 100).toFixed(2)
      }
      classes.push(c);
    }
    return classes;
  }

  /**
   * Decode out box coordinate
   *
   * @param {number[4]} offsets - An 4 element array of box shape offsets.
   * @param {number[4]} anchorCord - An 4 element array of anchor coordinate.
   */
  box_decode(offsets, anchorCord) {
    if (offsets.length !== 4 || anchorCord.length !== 4) {
      throw new Error('[box_decode] Each input length should be 4!');
    }
    const [ycenter_a, xcenter_a, ha, wa] = anchorCord;
    let [ty, tx, th, tw] = offsets;

    // scale_factors = [y_scale, x_scale, height_scale, width_scale]
    scale_factors = [10.0, 10.0, 5.0, 5.0];
    ty /= scale_factors[0];
    tx /= scale_factors[1];
    th /= scale_factors[2];
    tw /= scale_factors[3];

    let w = Math.exp(tw) * wa;
    let h = Math.exp(th) * ha;
    let ycenter = ty * ha + ycenter_a;
    let xcenter = tx * wa + xcenter_a;
    let ymin = ycenter - h / 2;
    let xmin = xcenter - w / 2;
    let ymax = ycenter + h / 2;
    let xmax = xcenter + w / 2;
    return [ymin, xmin, ymax, xmax];
  }

  /**
   * Get IOU(intersection-over-union) of 2 boxes
   *
   * @param {number[4]} boxCord1 - An 4 element array of box coordinate.
   * @param {number[4]} boxCord2 - An 4 element array of box coordinate.
   */
  IOU(boxCord1, boxCord2) {
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
   *
   */
  anchors(options) {
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
            anchors.push([y_center, x_center, height, width])
          });
        }
      }
    }
    console.log('box_specs_list', box_specs_list)
    console.log('anchors', anchors)
  }
}

function getOS() {
  var userAgent = window.navigator.userAgent,
      platform = window.navigator.platform,
      macosPlatforms = ['Macintosh', 'MacIntel', 'MacPPC', 'Mac68K'],
      windowsPlatforms = ['Win32', 'Win64', 'Windows', 'WinCE'],
      iosPlatforms = ['iPhone', 'iPad', 'iPod'],
      os = null;

  if (macosPlatforms.indexOf(platform) !== -1) {
    os = 'Mac OS';
  } else if (iosPlatforms.indexOf(platform) !== -1) {
    os = 'iOS';
  } else if (windowsPlatforms.indexOf(platform) !== -1) {
    os = 'Windows';
  } else if (/Android/.test(userAgent)) {
    os = 'Android';
  } else if (!os && /Linux/.test(platform)) {
    os = 'Linux';
  }

  return os;
}

function getNativeAPI() {
  const apiMapping = {
    'Mac OS': 'MPS',
    'Android': 'NN',
    'Windows': 'DirectML',
    'Linux': 'N/A'
  };

  return apiMapping[getOS()];
}

function getUrlParams( prop ) {
  var params = {};
  var search = decodeURIComponent( window.location.href.slice( window.location.href.indexOf( '?' ) + 1 ) );
  var definitions = search.split( '&' );

  definitions.forEach( function( val, key ) {
    var parts = val.split( '=', 2 );
      params[ parts[ 0 ] ] = parts[ 1 ];
  } );

  return ( prop && prop in params ) ? params[ prop ] : params;
}