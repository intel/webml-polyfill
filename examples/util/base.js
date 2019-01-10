let nnPolyfill, nnNative;
if (navigator.ml.isPolyfill) {
  nnNative = null;
  nnPolyfill = navigator.ml.getNeuralNetworkContext();
} else {
  nnNative = navigator.ml.getNeuralNetworkContext();
  nnPolyfill = navigator.ml_polyfill.getNeuralNetworkContext();
}

const currentOS = getOS();

const preferMap = {
  'MPS': 'sustained',
  'BNNS': 'fast',
  'sustained': 'SUSTAINED_SPEED',
  'fast': 'FAST_SINGLE_ANSWER',
  'low': 'LOW_POWER',
};

const mobilenet_v1_tflite = {
  modelName: 'Mobilenet V1(TFlite)',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/mobilenet_v1_1.0_224.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};

const mobilenet_v2_tflite = {
  modelName: 'Mobilenet V2(TFlite)',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/mobilenet_v2_1.0_224.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};

const inception_v3_tflite = {
  modelName: 'Inception V3(TFlite)',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/inception_v3.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
};

const inception_v4_tflite = {
  modelName: 'Inception V4(TFlite)',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/inception_v4.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};

const squeezenet_tflite = {
  modelName: 'Squeezenet(TFlite)',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/squeezenet.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};

const inception_resnet_v2_tflite = {
  modelName: 'Incep. Res. V2(TFlite)',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/inception_resnet_v2.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  postOptions: {
    softmax: true,
  }
};

const squeezenet_onnx = {
  modelName: 'SqueezeNet(Onnx)',
  modelFile: '../image_classification/model/squeezenet1.1.onnx',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // https://github.com/onnx/models/tree/master/models/image_classification/squeezenet#preprocessing
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    norm: true
  },
  postOptions: {
    softmax: true,
  }
};

const mobilenet_v2_onnx = {
  modelName: 'Mobilenet v2(Onnx)',
  modelFile: '../image_classification/model/mobilenetv2-1.0.onnx',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // https://github.com/onnx/models/tree/master/models/image_classification/mobilenet#preprocessing
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    norm: true
  },
  postOptions: {
    softmax: true,
  }
};

const resnet_v1_onnx = {
  modelName: 'ResNet50 v1(Onnx)',
  modelFile: '../image_classification/model/resnet50v1.onnx',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // https://github.com/onnx/models/tree/master/models/image_classification/resnet#preprocessing
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    norm: true
  },
  postOptions: {
    softmax: true,
  }
};

const resnet_v2_onnx = {
  modelName: 'ResNet50 v2(Onnx)',
  modelFile: '../image_classification/model/resnet50v2.onnx',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // https://github.com/onnx/models/tree/master/models/image_classification/resnet#preprocessing
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    norm: true
  },
  postOptions: {
    softmax: true,
  }
};

const inceptionv2_onnx = {
  modelName: 'Inception v2(Onnx)',
  modelFile: '../image_classification/model/inceptionv2.onnx',
  labelsFile: '../image_classification/model/ilsvrc2012labels.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
};

const densenet_onnx = {
  modelName: 'DenseNet(Onnx)',
  modelFile: '../image_classification/model/densenet121.onnx',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // mean and std should also be in BGR order
    mean: [0.406, 0.456, 0.485],
    std: [0.225, 0.224, 0.229],
    norm: true,
    channelScheme: 'BGR',
  },
  postOptions: {
    softmax: true,
  }
};

const ssd_mobilenetv1_tflite = {
  modelName: 'SSD MobileNetV1(TFlite)',
  modelFile: '../object_detection/model/ssd_mobilenet_v1.tflite',
  labelsFile: '../object_detection/model/coco_labels_list.txt',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};

const ssd_mobilenetv2_tflite = {
  modelName: 'SSD MobileNetV2(TFlite)',
  modelFile: '../object_detection/model/ssd_mobilenet_v2.tflite',
  labelsFile: '../object_detection/model/coco_labels_list.txt',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};

const ssdlite_mobilenetv2_tflite = {
  modelName: 'SSDLite MobileNetV2(TFlite)',
  modelFile: '../object_detection/model/ssdlite_mobilenet_v2.tflite',
  labelsFile: '../object_detection/model/coco_labels_list.txt',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};

const posenet = {
  modelName: 'PoseNet',
  inputSize: [513, 513, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};

const imageClassificationModels = [
  mobilenet_v1_tflite,
  mobilenet_v2_tflite,
  inception_v3_tflite,
  inception_v4_tflite,
  squeezenet_tflite,
  inception_resnet_v2_tflite,
  squeezenet_onnx,
  mobilenet_v2_onnx,
  resnet_v1_onnx,
  resnet_v2_onnx,
  inceptionv2_onnx,
  densenet_onnx,
];

const objectDetectionModels = [
  ssd_mobilenetv1_tflite,
  ssd_mobilenetv2_tflite,
  ssdlite_mobilenetv2_tflite,
];

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

function getNativeAPI(preferString) {
  const apiMapping = {
    'Android': {
      'sustained': 'NN',
      'fast': 'NN',
      'low': 'NN',
    },
    'Windows': {
      'sustained': 'clDNN',
      // 'fast': 'MKL-DNN', // implementing
    },
    'Linux': {
      'sustained': 'clDNN',
      // 'fast': 'MKL-DNN', // implementing
    },
    'Mac OS': {
      'fast': 'BNNS',
      'sustained': 'MPS',
    }
  };
  return apiMapping[currentOS][preferString];
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

function getPreferParam() {
  // workaround for using MPS backend on Mac OS by visiting URL with 'prefer=sustained'
  // workaround for using BNNS backend on Mac OS by visiting URL with 'prefer=fast'
  // use 'sustained' as default for Mac OS
  var prefer = 'sustained';
  var parameterStr = window.location.search.substr(1);
  var reg = new RegExp("(^|&)prefer=([^&]*)(&|$)", "i");
  var r = parameterStr.match(reg);
  if (r != null) {
    prefer = unescape(r[2]);
    if (prefer !== 'fast' && prefer !== 'sustained') {
      prefer = 'invalid';
    }
  }

  return prefer;
}

function getPrefer(backend) {
  let nn = navigator.ml.getNeuralNetworkContext();
  let prefer = nn.PREFER_FAST_SINGLE_ANSWER;
  if (currentOS === 'Mac OS' && backend === 'WebML') {
      let urlPrefer = getPreferParam();
      if (urlPrefer === 'sustained') {
        prefer = nn.PREFER_SUSTAINED_SPEED;
      } else if (urlPrefer === 'fast') {
        prefer = nn.PREFER_FAST_SINGLE_ANSWER;
      }
  }
  return prefer;
}

function getPreferCode(backend, prefer) {
  let preferCode;
  let nn = navigator.ml.getNeuralNetworkContext();
  if (backend === 'WASM') {
    preferCode = nn.PREFER_FAST_SINGLE_ANSWER;
  } else if (backend === 'WebGL') {
    preferCode = nn.PREFER_SUSTAINED_SPEED;
  } else if (backend === 'WebML') {
    if (prefer === 'sustained') {
      preferCode = nn.PREFER_SUSTAINED_SPEED;
    } else if (prefer === 'fast') {
      preferCode = nn.PREFER_FAST_SINGLE_ANSWER;
    } else if (prefer === 'low') {
      preferCode = nn.PREFER_LOW_POWER;
    }
  }
  return preferCode;
}
