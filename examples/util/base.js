let nnPolyfill, nnNative;
if (navigator.ml.isPolyfill) {
  nnNative = null;
  nnPolyfill = navigator.ml.getNeuralNetworkContext();
} else {
  nnNative = navigator.ml.getNeuralNetworkContext();
  nnPolyfill = navigator.ml_polyfill.getNeuralNetworkContext();
}

const preferMap = {
  'MPS': 'sustained',
  'BNNS': 'fast',
  'sustained': 'SUSTAINED_SPEED',
  'fast': 'FAST_SINGLE_ANSWER',
  'low': 'LOW_POWER',
};


const imageClassificationModels = [{
  modelName: 'Mobilenet v1 (TFLite)',
  modelFormatName: 'mobilenet_v1_tflite',
  modelSize: '16.9MB',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/mobilenet_v1_1.0_224.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
  paperUrl: 'https://arxiv.org/pdf/1704.04861.pdf'
}, {
  modelName: 'Mobilenet v2 (TFLite)',
  modelFormatName: 'mobilenet_v2_tflite',
  modelSize: '14.0MB',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/mobilenet_v2_1.0_224.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'MobileNetV2 improves the state of the art performance of mobile models.',
  paperUrl: 'https://arxiv.org/abs/1801.04381'
}, {
  modelName: 'Inception v3 (TFLite)',
  modelFormatName: 'inception_v3_tflite',
  modelSize: '95.3MB',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/inception_v3.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'Inception-v3 is trained for the ImageNet Large Visual Recognition Challenge.',
  paperUrl: 'http://arxiv.org/abs/1512.00567'
}, {
  modelName: 'Inception v4 (TFLite)',
  modelFormatName: 'inception_v4_tflite',
  modelSize: '170.7MB',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/inception_v4.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'Inception architecture that has been shown to achieve very good performance at relatively low computational cost.',
  paperUrl: 'https://arxiv.org/abs/1602.07261'
}, {
  modelName: 'Squeezenet (TFLite)',
  modelFormatName: 'squeezenet_tflite',
  modelSize: '5.0MB',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/squeezenet.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters.',
  paperUrl: 'https://arxiv.org/abs/1602.07360'
}, {
  modelName: 'Inception Resnet v2 (TFLite)',
  modelFormatName: 'inception_resnet_v2_tflite',
  modelSize: '121.0MB',
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
  },
  intro: 'Inception architecture that has been shown to achieve very good performance at relatively low computational cost, and training with residual connections accelerates the training of Inception networks significantly. There is also some evidence of residual Inception networks outperforming similarly expensive Inception networks without residual connections.',
  paperUrl: 'https://arxiv.org/abs/1602.07261'
}, {
  modelName: 'Squeezenet (ONNX)',
  modelFormatName: 'squeezenet_onnx',
  modelSize: '5.0MB',
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
  },
  intro: 'A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters.',
  paperUrl: 'https://arxiv.org/abs/1602.07360'
}, {
  modelName: 'Mobilenet v2 (ONNX)',
  modelFormatName: 'mobilenet_v2_onnx',
  modelSize: '14.2MB',
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
  },
  intro: 'MobileNetV2 improves the state of the art performance of mobile models.',
  paperUrl: 'https://arxiv.org/abs/1801.04381'
}, {
  modelName: 'Resnet v1 (ONNX)',
  modelFormatName: 'resnet_v1_onnx',
  modelSize: '102.6MB',
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
  },
  intro: 'A residual learning framework to ease the training of networks that are substantially deeper than those used previously. This result won the 1st place on the ILSVRC 2015 classification task.',
  paperUrl: 'https://arxiv.org/abs/1512.03385'
}, {
  modelName: 'Resnet v2 (ONNX)',
  modelFormatName: 'resnet_v2_onnx',
  modelSize: '102.4MB',
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
  },
  intro: 'Deep residual networks have emerged as a family of extremely deep architectures showing compelling accuracy and nice convergence behaviors. It reports improved results using a 1001-layer ResNet on CIFAR-10 (4.62% error) and CIFAR-100, and a 200-layer ResNet on ImageNet.',
  paperUrl: 'https://arxiv.org/abs/1603.05027'
}, {
  modelName: 'Inception v2 (ONNX)',
  modelFormatName: 'inception_v2_onnx',
  modelSize: '45.0MB',
  modelFile: '../image_classification/model/inceptionv2.onnx',
  labelsFile: '../image_classification/model/ilsvrc2012labels.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  intro: 'Inception-v2 is trained for the ImageNet Large Visual Recognition Challenge.',
  paperUrl: 'https://arxiv.org/abs/1512.00567'
}, {
  modelName: 'Densenet (ONNX)',
  modelFormatName: 'densenet_onnx',
  modelSize: '32.7MB',
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
  },
  intro: 'Dense Convolutional Network (DenseNet) connects each layer to every other layer in a feed-forward fashion. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. ',
  paperUrl: 'https://arxiv.org/abs/1608.06993'
}];

const objectDetectionModels = [{
  modelName: 'SSD MobileNet v1 (TFLite)',
  modelFormatName: 'ssd_mobilenet_v1_tflite',
  modelSize: '27.3MB',
  modelFile: '../object_detection/model/ssd_mobilenet_v1.tflite',
  labelsFile: '../object_detection/model/coco_labels_list.txt',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'SSD (Single Shot MultiBox Detector) is an unified framework for object detection with a single network. Loading SSD MobileNet model (converted from Tensorflow SSD MobileNet model) trained by COCO in TensorFlow Lite format, constructs and inferences it by WebML API.',
  paperUrl: 'https://arxiv.org/abs/1803.08225'
}, {
  modelName: 'SSD MobileNet v2 (TFLite)',
  modelFormatName: 'ssd_mobilenet_v2_tflite',
  modelSize: '67.3MB',
  modelFile: '../object_detection/model/ssd_mobilenet_v2.tflite',
  labelsFile: '../object_detection/model/coco_labels_list.txt',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'MobileNet V2 with SSD. This model is faster than Mobilenet V1 SSD at the same accuracy.',
  paperUrl: 'https://arxiv.org/abs/1801.04381'
}, {
  modelName: 'SSDLite MobileNet v2 (TFLite)',
  modelFormatName: 'ssdlite_mobilenet_v2_tflite',
  modelSize: '17.9MB',
  modelFile: '../object_detection/model/ssdlite_mobilenet_v2.tflite',
  labelsFile: '../object_detection/model/coco_labels_list.txt',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'MobileNet V2 with SSDLite. This model is faster than Mobilenet V1 SSD at the same accuracy.',
  paperUrl: 'https://arxiv.org/abs/1801.04381'
}];

const humanPoseEstimationModels = [{
  modelName: 'PoseNet',
  modelFormatName: 'posenet',
  modelSize: '13.3MB',
  modelFile: '../skeleton_detection/model/mobilenet_v1_101',
  inputSize: [513, 513, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'PoseNet is a machine learning model that allows for Real-time Human Pose Estimation which can be used to estimate either a single pose or multiple poses.',
  paperUrl: 'https://arxiv.org/abs/1803.08225'
}];

const semanticSegmentationModels = [{
    modelName: 'Deeplab 224 (TFLite)',
    modelFormatName: 'deeplab_mobilenet_v2_224_tflite',
    modelSize: '9.5MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_224.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [224, 224, 3],
    outputSize: [224, 224, 21],
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 224 Atrous (TFLite)',
    modelFormatName: 'deeplab_mobilenet_v2_224_atrous_tflite',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_224_dilated.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [224, 224, 3],
    outputSize: [224, 224, 21],
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 257 (TFLite)',
    modelFormatName: 'deeplab_mobilenet_v2_257_tflite',
    modelSize: '9.5MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_257.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [257, 257, 3],
    outputSize: [257, 257, 21],
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 257 Atrous (TFLite)',
    modelFormatName: 'deeplab_mobilenet_v2_257_atrous_tflite',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_257_dilated.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [257, 257, 3],
    outputSize: [257, 257, 21],
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 321 (TFLite)',
    modelFormatName: 'deeplab_mobilenet_v2_321_tflite',
    modelSize: '9.5MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_321.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [321, 321, 3],
    outputSize: [321, 321, 21],
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 321 Atrous (TFLite)',
    modelFormatName: 'deeplab_mobilenet_v2_321_atrous_tflite',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_321_dilated.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [321, 321, 3],
    outputSize: [321, 321, 21],
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 513 (TFLite)',
    modelFormatName: 'deeplab_mobilenet_v2_513_tflite',
    modelSize: '9.5MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_513.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 21],
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 513 Atrous (TFLite)',
    modelFormatName: 'deeplab_mobilenet_v2_513_atrous_tflite',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_513_dilated.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 21],
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
}];

const getOS = () => {
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

const currentOS = getOS();

const getNativeAPI = (preferString) => {
  const apiMapping = {
    'Android': {
      'sustained': 'NN',
      'fast': 'NN',
      'low': 'NN',
    },
    'Windows': {
      'sustained': 'clDNN',
      'fast': 'mklDNN',
    },
    'Linux': {
      'sustained': 'clDNN',
      'fast': 'mklDNN',
    },
    'Mac OS': {
      'fast': 'BNNS',
      'sustained': 'MPS',
    }
  };
  return apiMapping[currentOS][preferString];
}

const getUrlParams = (prop) => {
  var params = {};
  var search = decodeURIComponent(window.location.href.slice(window.location.href.indexOf('?') + 1));
  var definitions = search.split('&');

  definitions.forEach((val, key) => {
    var parts = val.split('=', 2);
    params[parts[0]] = parts[1];
  });

  return (prop && prop in params) ? params[prop] : params;
}

const getPreferParam = () => {
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

const getPrefer = (backend) => {
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

const getPreferCode = (backend, prefer) => {
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

const getSearchParamsPrefer = () => {
  let searchParams = new URLSearchParams(location.search);
  return searchParams.has('prefer') ? searchParams.get('prefer') : '';
}

const getSearchParamsBackend = () => {
  let searchParams = new URLSearchParams(location.search);
  return searchParams.has('b') ? searchParams.get('b') : '';
}
const getSearchParamsModel = () => {
  let searchParams = new URLSearchParams(location.search);
  if (searchParams.has('m') && searchParams.has('t')) {
    return searchParams.get('m') + '_' + searchParams.get('t');
  } else {
    return '';
  }
}
