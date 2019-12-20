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

const modelZoo = {

imageClassificationModels: [{
  modelName: 'MobileNet v1 (TFLite)',
  format: 'TFLite',
  modelId: 'mobilenet_v1_tflite',
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
  modelName: 'MobileNet v1 Quant (TFLite)',
  format: 'TFLite',
  modelId: 'mobilenet_v1_quant_tflite',
  isQuantized: true,
  modelSize: '4.3MB',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/mobilenet_v1_1.0_224_quant.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  intro: 'Quantized version of Mobilenet v1',
  paperUrl: 'https://arxiv.org/pdf/1712.05877.pdf'
}, {
  modelName: 'MobileNet v2 (TFLite)',
  format: 'TFLite',
  modelId: 'mobilenet_v2_tflite',
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
  modelName: 'MobileNet v2 Quant (TFLite)',
  format: 'TFLite',
  modelId: 'mobilenet_v2_quant_tflite',
  isQuantized: true,
  modelSize: '6.9MB',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/mobilenet_v2_1.0_224_quant.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  postOptions: {
    softmax: true,
  },
  intro: 'Quantized version of Mobilenet v2',
  paperUrl: 'https://arxiv.org/abs/1806.08342'
}, {
  modelName: 'SqueezeNet (TFLite)',
  format: 'TFLite',
  modelId: 'squeezenet_tflite',
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
  modelName: 'Inception v3 (TFLite)',
  format: 'TFLite',
  modelId: 'inception_v3_tflite',
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
  modelName: 'Inception v3 Quant (TFLite)',
  format: 'TFLite',
  modelId: 'inception_v3_quant_tflite',
  isQuantized: true,
  modelSize: '23.9MB',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/inception_v3_quant.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  postOptions: {
    softmax: true,
  },
  intro: 'Quantized version of Inception v3.',
  paperUrl: 'https://arxiv.org/abs/1806.08342'
}, {
  modelName: 'Inception v4 (TFLite)',
  format: 'TFLite',
  modelId: 'inception_v4_tflite',
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
  modelName: 'Inception v4 Quant (TFLite)',
  format: 'TFLite',
  modelId: 'inception_v4_quant_tflite',
  isQuantized: true,
  modelSize: '42.9MB',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: '../image_classification/model/inception_v4_299_quant.tflite',
  labelsFile: '../image_classification/model/labels1001.txt',
  intro: 'Quantized version of Inception v4.',
  paperUrl: 'https://arxiv.org/abs/1602.07261'
}, {
  modelName: 'Inception ResNet v2 (TFLite)',
  format: 'TFLite',
  modelId: 'inception_resnet_v2_tflite',
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
  modelName: 'SqueezeNet (ONNX)',
  format: 'ONNX',
  modelId: 'squeezenet_onnx',
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
  modelName: 'MobileNet v2 (ONNX)',
  format: 'ONNX',
  modelId: 'mobilenet_v2_onnx',
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
  modelName: 'ResNet50 v1 (ONNX)',
  format: 'ONNX',
  modelId: 'resnet_v1_onnx',
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
  modelName: 'ResNet50 v2 (ONNX)',
  format: 'ONNX',
  modelId: 'resnet_v2_onnx',
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
  format: 'ONNX',
  modelId: 'inception_v2_onnx',
  modelSize: '45.0MB',
  modelFile: '../image_classification/model/inceptionv2.onnx',
  labelsFile: '../image_classification/model/ilsvrc2012labels.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  intro: 'Inception-v2 is trained for the ImageNet Large Visual Recognition Challenge.',
  paperUrl: 'https://arxiv.org/abs/1512.00567'
}, {
  modelName: 'DenseNet 121 (ONNX)',
  format: 'ONNX',
  modelId: 'densenet_121_onnx',
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
}, {
  modelName: 'SqueezeNet (OpenVino)',
  format: 'OpenVINO',
  modelId: 'squeezenet_openvino',
  modelSize: '4.9MB',
  modelFile: '../image_classification/model/squeezenet1.1.bin',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [227, 227, 3],
  outputSize: 1000,
  preOptions: {
    channelScheme: 'BGR',
  },
  intro: 'A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters.',
  paperUrl: 'https://arxiv.org/abs/1602.07360'
}, {
  modelName: 'MobileNet v1 (OpenVino)',
  format: 'OpenVINO',
  modelId: 'mobilenet_v1_openvino',
  modelSize: '16.9MB',
  modelFile: '../image_classification/model/mobilenet_v1_1.0_224.bin',
  labelsFile: '../image_classification/model/labels1001.txt',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
  paperUrl: 'https://arxiv.org/pdf/1704.04861.pdf'
}, {
  modelName: 'MobileNet v2 (OpenVino)',
  format: 'OpenVINO',
  modelId: 'mobilenet_v2_openvino',
  modelSize: '14.0MB',
  modelFile: '../image_classification/model/mobilenet_v2_1.0_224.bin',
  labelsFile: '../image_classification/model/labels1001.txt',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'MobileNetV2 improves the state of the art performance of mobile models.',
  paperUrl: 'https://arxiv.org/abs/1801.04381'
}, {
  modelName: 'ResNet50 v1 (OpenVino)',
  format: 'OpenVINO',
  modelId: 'resnet50_v1_openvino',
  modelSize: '102.1MB',
  modelFile: '../image_classification/model/resnet-50.bin',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    channelScheme: 'BGR',
  },
  intro: 'A residual learning framework to ease the training of networks that are substantially deeper than those used previously. This result won the 1st place on the ILSVRC 2015 classification task.',
  paperUrl: 'https://arxiv.org/abs/1512.03385'
}, {
  modelName: 'DenseNet 121 (OpenVino)',
  format: 'OpenVINO',
  modelId: 'densenet_121_openvino',
  modelSize: '31.9MB',
  modelFile: '../image_classification/model/densenet-121.bin',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
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
}, {
  modelName: 'Inception v2 (OpenVino)',
  format: 'OpenVINO',
  modelId: 'inception_v2_openvino',
  modelSize: '44.7MB',
  modelFile: '../image_classification/model/googlenet-v2.bin',
  labelsFile: '../image_classification/model/ilsvrc2012labels.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    channelScheme: 'BGR',
  },
  intro: 'Inception-v2 is trained for the ImageNet Large Visual Recognition Challenge.',
  paperUrl: 'https://arxiv.org/abs/1512.00567'
}, {
  modelName: 'Inception v4 (OpenVino)',
  format: 'OpenVINO',
  modelId: 'inception_v4_openvino',
  modelSize: '170.6MB',
  modelFile: '../image_classification/model/googlenet-v4.bin',
  labelsFile: '../image_classification/model/labels1000.txt',
  inputSize: [299, 299, 3],
  outputSize: 1000,
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'Inception architecture that has been shown to achieve very good performance at relatively low computational cost.',
  paperUrl: 'https://arxiv.org/abs/1602.07261'
}],

objectDetectionModels: [{
  modelName: 'SSD MobileNet v1 (TFLite)',
  format: 'TFLite',
  modelId: 'ssd_mobilenet_v1_tflite',
  modelSize: '27.3MB',
  modelFile: '../object_detection/model/ssd_mobilenet_v1.tflite',
  labelsFile: '../object_detection/model/coco_classes.txt',
  type: 'SSD',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  margin: [1, 1, 1, 1],
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'SSD (Single Shot MultiBox Detector) is an unified framework for object detection with a single network. Loading SSD MobileNet model (converted from Tensorflow SSD MobileNet model) trained by COCO in TensorFlow Lite format, constructs and inferences it by WebML API.',
  paperUrl: 'https://arxiv.org/abs/1803.08225'
}, {
  modelName: 'SSD MobileNet v1 Quant (TFLite)',
  format: 'TFLite',
  modelId: 'ssd_mobilenet_v1_quant_tflite',
  isQuantized: true,
  modelSize: '6.9MB',
  modelFile: '../object_detection/model/ssd_mobilenet_v1_quant.tflite',
  labelsFile: '../object_detection/model/coco_classes.txt',
  type: 'SSD',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  margin: [1, 1, 1, 1],
  inputSize: [300, 300, 3],
  intro: 'Quantized version of SSD Mobilenet v1',
  paperUrl: 'https://arxiv.org/pdf/1712.05877.pdf'
}, {
  modelName: 'SSD MobileNet v2 (TFLite)',
  format: 'TFLite',
  modelId: 'ssd_mobilenet_v2_tflite',
  modelSize: '67.3MB',
  modelFile: '../object_detection/model/ssd_mobilenet_v2.tflite',
  labelsFile: '../object_detection/model/coco_classes.txt',
  type: 'SSD',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  margin: [1, 1, 1, 1],
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'SSD MobileNet V2 is slower than SSD Mobilenet V1, but has higher accuracy.',
  paperUrl: 'https://arxiv.org/abs/1801.04381'
}, {
  modelName: 'SSD MobileNet v2 Quant (TFLite)',
  format: 'TFLite',
  modelId: 'ssd_mobilenet_v2_quant_tflite',
  isQuantized: true,
  modelSize: '6.2MB',
  modelFile: '../object_detection/model/ssd_mobilenet_v2_quant.tflite',
  labelsFile: '../object_detection/model/coco_classes.txt',
  type: 'SSD',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  margin: [1, 1, 1, 1],
  inputSize: [300, 300, 3],
  intro: 'Quantized version of SSD Mobilenet v2',
  paperUrl: 'https://arxiv.org/abs/1806.08342'
}, {
  modelName: 'SSDLite MobileNet v2 (TFLite)',
  format: 'TFLite',
  modelId: 'ssdlite_mobilenet_v2_tflite',
  modelSize: '17.9MB',
  modelFile: '../object_detection/model/ssdlite_mobilenet_v2.tflite',
  labelsFile: '../object_detection/model/coco_classes.txt',
  type: 'SSD',
  box_size: 4,
  num_classes: 91,
  num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
  margin: [1, 1, 1, 1],
  inputSize: [300, 300, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'SSDLite MobileNet V2 is an upgraded version of SSD MobileNet V2. Compared with SSD Mobilenet V2, SSDLite Mobilenet V2 is much faster, and almost has no loss of the accuracy.',
  paperUrl: 'https://arxiv.org/abs/1801.04381'
}, {
  modelName: 'Tiny Yolo v2 COCO (TFLite)',
  format: 'TFLite',
  modelId: 'tiny_yolov2_coco_tflite',
  modelSize: '44.9MB',
  modelFile: '../object_detection/model/tiny_yolov2_coco.tflite',
  labelsFile: '../object_detection/model/coco_classes_part.txt',
  type: 'YOLO',
  num_classes: 80,
  margin: [1, 1, 1, 1],
  inputSize: [416, 416, 3],
  outputSize: 1 * 13 * 13 * 425,
  anchors: [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
  preOptions: {
    norm: true,
  },
  intro: 'Tiny YOLO is based off of the Darknet reference network and is much faster but less accurate than the normal YOLO model. And this model is trained by COCO dataset.',
  paperUrl: 'https://arxiv.org/abs/1612.08242'
}, {
  modelName: 'Tiny Yolo v2 VOC (TFLite)',
  format: 'TFLite',
  modelId: 'tiny_yolov2_voc_tflite',
  modelSize: '63.4MB',
  modelFile: '../object_detection/model/tiny_yolov2_voc.tflite',
  labelsFile: '../object_detection/model/pascal_classes.txt',
  type: 'YOLO',
  num_classes: 20,
  margin: [1, 1, 1, 1],
  inputSize: [416, 416, 3],
  outputSize: 1 * 13 * 13 * 125,
  anchors: [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
  preOptions: {
    norm: true,
  },
  intro: 'Tiny YOLO is based off of the Darknet reference network and is much faster but less accurate than the normal YOLO model. And this model is trained by VOC dataset.',
  paperUrl: 'https://arxiv.org/abs/1612.08242'
}],

humanPoseEstimationModels: [{
  modelName: 'PoseNet',
  format: '',
  modelId: 'posenet',
  modelSize: '13.3MB',
  modelFile: '../skeleton_detection/model/mobilenet_v1_101',
  inputSize: [513, 513, 3],
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  intro: 'PoseNet is a machine learning model that allows for Real-time Human Pose Estimation which can be used to estimate either a single pose or multiple poses.',
  paperUrl: 'https://arxiv.org/abs/1803.08225'
}],

semanticSegmentationModels: [{
    modelName: 'Deeplab 224 (TFLite)',
    format: 'TFLite',
    modelId: 'deeplab_mobilenet_v2_224_tflite',
    modelSize: '9.5MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_224.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [224, 224, 3],
    outputSize: [224, 224, 1],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 224 Atrous (TFLite)',
    format: 'TFLite',
    modelId: 'deeplab_mobilenet_v2_224_atrous_tflite',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_224_dilated.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [224, 224, 3],
    outputSize: [224, 224, 1],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 257 (TFLite)',
    format: 'TFLite',
    modelId: 'deeplab_mobilenet_v2_257_tflite',
    modelSize: '9.5MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_257.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [257, 257, 3],
    outputSize: [257, 257, 1],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 257 Atrous (TFLite)',
    format: 'TFLite',
    modelId: 'deeplab_mobilenet_v2_257_atrous_tflite',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_257_dilated.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [257, 257, 3],
    outputSize: [257, 257, 1],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 321 (TFLite)',
    format: 'TFLite',
    modelId: 'deeplab_mobilenet_v2_321_tflite',
    modelSize: '9.5MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_321.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [321, 321, 3],
    outputSize: [321, 321, 1],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 321 Atrous (TFLite)',
    format: 'TFLite',
    modelId: 'deeplab_mobilenet_v2_321_atrous_tflite',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_321_dilated.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [321, 321, 3],
    outputSize: [321, 321, 1],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 513 (TFLite)',
    format: 'TFLite',
    modelId: 'deeplab_mobilenet_v2_513_tflite',
    modelSize: '9.5MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_513.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 1],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab 513 Atrous (TFLite)',
    format: 'TFLite',
    modelId: 'deeplab_mobilenet_v2_513_atrous_tflite',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_513_dilated.tflite',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 1],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
}],

superResolutionModels: [
  {
    modelName: 'SRGAN 96x4 (TFLite)',
    format: 'TFLite',
    modelId: 'srgan_96x4_tflite',
    modelSize: '6.1MB',
    inputSize: [96, 96, 3],
    outputSize: [384, 384, 3],
    scale: 4,
    modelFile: '../super_resolution/model/srgan_96_4.tflite',
    intro: 'Photo-realistic single image Super-Resolution using a generative adversarial network.',
    paperUrl: 'https://arxiv.org/abs/1609.04802'
  },
  {
    modelName: 'SRGAN 128x4 (TFLite)',
    format: 'TFLite',
    modelId: 'srgan_128x4_tflite',
    modelSize: '6.1MB',
    inputSize: [128, 128, 3],
    outputSize: [512, 512, 3],
    scale: 4,
    modelFile: '../super_resolution/model/srgan_128_4.tflite',
    intro: 'Photo-realistic single image Super-Resolution using a generative adversarial network.',
    paperUrl: 'https://arxiv.org/abs/1609.04802'
  }
],

faceDetectionModels: [{
    modelName: 'SSD MobileNet v1 Face (TFlite)',
    format: 'TFLite',
    modelId: 'ssd_mobilenetv1_face_tflite',
    modelSize: '22.0MB',
    type: 'SSD',
    modelFile: '../facial_landmark_detection/model/ssd_mobilenetv1_face.tflite',
    box_size: 4,
    num_classes: 2,
    num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
    margin: [1.2,1.2,0.8,1.1],
    inputSize: [300, 300, 3],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'SSD Mobilenet V1 Face is based on SSD Mobilenet V1 model structure, and is trained by Tensorflow Object Detection API with WIDER_FACE dataset for face detection task.',
    paperUrl: 'https://arxiv.org/abs/1803.08225'
  }, {
    modelName: 'SSD MobileNet v2 Face (TFlite)',
    format: 'TFLite',
    modelId: 'ssd_mobilenetv2_face_tflite',
    modelSize: '18.4MB',
    type: 'SSD',
    modelFile: '../facial_landmark_detection/model/ssd_mobilenetv2_face.tflite',
    box_size: 4,
    num_classes: 2,
    num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
    margin: [1.2,1.2,0.8,1.1],
    inputSize: [300, 300, 3],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'SSD Mobilenet V2 Face is based on SSD Mobilenet V2 model structure, and is trained by Tensorflow Object Detection API with WIDER_FACE dataset for face detection task.',
    paperUrl: 'https://arxiv.org/abs/1801.04381'
  }, {
    modelName: 'SSDLite MobileNet v2 Face (TFlite)',
    format: 'TFLite',
    modelId: 'ssdlite_mobilenetv2_face_tflite',
    modelSize: '12.1MB',
    type: 'SSD',
    modelFile: '../facial_landmark_detection/model/ssdlite_mobilenetv2_face.tflite',
    box_size: 4,
    num_classes: 2,
    num_boxes: 1083 + 600 + 150 + 54 + 24 + 6,
    margin: [1.2,1.2,0.8,1.1],
    inputSize: [300, 300, 3],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'SSDLite Mobilenet V2 Face is based on SSDLite Mobilenet V2 model structure, and is trained by Tensorflow Object Detection API with WIDER_FACE dataset for face detection task.',
    paperUrl: 'https://arxiv.org/abs/1801.04381'
  }, {
    modelName: 'Tiny Yolo v2 Face (TFlite)',
    format: 'TFLite',
    modelId: 'tiny_yolov2_face_tflite',
    modelSize: '44.1MB',
    modelFile: '../facial_landmark_detection/model/tiny_yolov2_face.tflite',
    type: 'YOLO',
    margin: [1.15, 1.15, 0.6, 1.15],
    inputSize: [416, 416, 3],
    outputSize: 1 * 13 * 13 * 30,
    anchors: [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
    preOptions: {
      norm: true,
    },
    intro: 'Tiny YOLO V2 Face is based off the Darknet reference network and trained with WIDER_FACE dataset for face detection task.',
    paperUrl: 'https://arxiv.org/abs/1612.08242'
}],

facialLandmarkDetectionModels: [{
  modelName: 'SimpleCNN (TFlite)',
  format: 'TFLite',
  modelId: 'face_landmark_tflite',
  modelSize: '29.4MB',
  modelFile: '../facial_landmark_detection/model/face_landmark.tflite',
  inputSize: [128, 128, 3],
  outputSize: 136,
  intro: 'A simple CNN model uses regression methods to map human facial features to 68 key points.',
  paperUrl: 'https://www.sciencedirect.com/science/article/pii/S0262885615001341'
}],

emotionAnalysisModels: [{
  modelName: 'Simple CNN 7 (TFlite)',
  format: 'TFLite',
  modelId: 'emotion_analysis_tflite',
  modelSize: '7.3MB',
  modelFile: '../emotion_analysis/model/emotion_classification_7.tflite',
  labels: ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
  inputSize: [48, 48, 1],
  outputSize: 7,
  preOptions: {
    norm: true,
  },
  intro: 'A simple CNN model uses regression methods to map human facial features to seven different types of emotion class.',
  paperUrl: ''
}],

speechCommandModels: [{
  modelName: 'KWS CNN (TFLite)',
  format: 'TFLite',
  modelId: 'kws_cnn_tflite',
  modelSize: '3.4MB',
  inputSize: [1, 16000],
  outputSize: [1, 1, 12],
  sampleRate: 16000,
  modelFile: '../speech_commands/model/kws_cnn.tflite',
  labelsFile: '../speech_commands/model/labels.txt',
  intro: 'Use Convolutional Neural Networks (CNNs) for a small-footprint keyword spotting (KWS) task.',
  paperUrl: 'https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf'
}],

faceRecognitionModels: [{
    modelName: 'FaceNet (OpenVINO)',
    format: 'OpenVino',
    modelId: 'facenet_recognition_openvino',
    modelSize: '93.9MB',
    modelFile: '../face_recognition/model/facenet.bin',
    inputSize: [1, 160, 160, 3],
    outputSize: [1, 512],
    preOptions: {
      mean: [0, 0, 0, 0],
      std: [1, 1, 1, 1],
      channelScheme: 'BGR',
      format: 'NHWC'
    },
    postOptions: {
      distanceMetric: 'euclidean',
      threshold: 1.26
    },
    intro: 'This is a CNN model for face recognition which learns discriminative features of faces and produces embeddings for input face images.',
    paperUrl: 'https://arxiv.org/abs/1503.03832'
  }, {
    modelName: 'Face Reidentification (OpenVINO)',
    format: 'OpenVino',
    modelId: 'facereidentification_recognition_openvino',
    modelSize: '4.4MB',
    modelFile: '../face_recognition/model/face-reidentification-retail-0095.bin',
    inputSize: [1, 128, 128, 3],
    outputSize: [1, 1, 1, 256],
    preOptions: {
      mean: [0, 0, 0, 0],
      std: [1, 1, 1, 1],
      channelScheme: 'BGR',
      format: 'NHWC'
    },
    postOptions: {
      distanceMetric: 'cosine',
      threshold: 0.8
    },
    intro: 'This is a lightweight network for the face re-identification scenario that is\
      based on MobileNet V2. The model produces feature vectors which should be close in cosine distance for similar faces and far for different faces.'
}]

};

// extract model lists into the browser env for backward compatiblity
for (const modelListName in modelZoo) {
  window[modelListName] = modelZoo[modelListName];
}

const getModelById = (id) => {
  for (const model of Object.values(modelZoo).flat()) {
    if (id === model.modelId) {
      return model;
    }
  }
  return {};
};

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
let eager = false;
let supportedOps = new Set();

const getNativeAPI = (preferString) => {
  // if you are going to modify the backend name, please change the
  // `backendEnums` in the `getDefaultSupportedOps` below
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
      'low': 'Myriad',
    },
    'Mac OS': {
      'fast': 'BNNS',
      'sustained': 'MPS',
    }
  };
  return apiMapping[currentOS][preferString];
}

const getDefaultSupportedOps = (backend, prefer) => {
  if (prefer === 'none' && backend !== 'WebML') {
    // if `prefer` is none, all ops should only run in polyfill
    return new Set();
  }

  // backend enums are defined in the `getNativeAPI` above
  const backendEnums =        { NN: 0,    MPS: 1,  BNNS: 2,  clDNN: 3, mklDNN: 4, DirectML: 5 };
  const supportedTable = {
    ADD:                      [ true,     true,    true,     true,     false,    true  ],
    ATROUS_CONV_2D:           [ false,    false,   false,    true,     true,     true  ],
    ATROUS_DEPTHWISE_CONV_2D: [ false,    false,   false,    true,     true,     true  ],
    AVERAGE_POOL_2D:          [ true,     true,    true,     true,     true,     true  ],
    CONCATENATION:            [ true,     true,    true,     true,     true,     true  ],
    CONV_2D:                  [ true,     true,    true,     true,     true,     true  ],
    DEPTHWISE_CONV_2D:        [ true,     true,    false,    true,     true,     true  ],
    FULLY_CONNECTED:          [ true,     true,    true,     true,     true,     true  ],
    MAX_POOL_2D:              [ true,     true,    true,     true,     true,     true  ],
    MUL:                      [ true,     true,    true,     true,     false,    true  ],
    RESHAPE:                  [ true,     true,    true,     true,     true,     true  ],
    RESIZE_BILINEAR:          [ true,     false,   true,     true,     false,    false ],
    SOFTMAX:                  [ true,     true,    true,     true,     true,     true  ],
    ARGMAX:                   [ false,    true,    false,    true,     false,    true  ]
  };

  const nn = navigator.ml.getNeuralNetworkContext();
  const supportedOps = new Set();
  const backendId = backendEnums[getNativeAPI(prefer)];
  for (const opName in supportedTable) {
    if (supportedTable[opName][backendId]) {
      supportedOps.add(nn[opName]);
    }
  }
  return supportedOps;
};

const operationTypes = {
   // Operation types.
   0: 'ADD',
   1: 'AVERAGE_POOL_2D',
   2: 'CONCATENATION',
   3: 'CONV_2D',
   4: 'DEPTHWISE_CONV_2D',
   5: 'DEPTH_TO_SPACE',
   6: 'DEQUANTIZE',
   7: 'EMBEDDING_LOOKUP',
   8: 'FLOOR',
   9: 'FULLY_CONNECTED',
   10: 'HASHTABLE_LOOKUP',
   11: 'L2_NORMALIZATION',
   12: 'L2_POOL_2D',
   13: 'LOCAL_RESPONSE_NORMALIZATION',
   14: 'LOGISTIC',
   15: 'LSH_PROJECTION',
   16: 'LSTM',
   17: 'MAX_POOL_2D',
   18: 'MUL',
   19: 'RELU',
   20: 'RELU1',
   21: 'RELU6',
   22: 'RESHAPE',
   23: 'RESIZE_BILINEAR',
   24: 'RNN',
   25: 'SOFTMAX',
   26: 'SPACE_TO_DEPTH',
   27: 'SVDF',
   28: 'TANH',
   29: 'BATCH_TO_SPACE_ND',
   37: 'TRANSPOSE',
   39: 'ARGMAX',
   65: 'MAXIMUM',
   10003: 'ATROUS_CONV_2D',
   10004: 'ATROUS_DEPTHWISE_CONV_2D'
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
  if (prefer === 'sustained') {
    preferCode = nn.PREFER_SUSTAINED_SPEED;
  } else if (prefer === 'fast') {
    preferCode = nn.PREFER_FAST_SINGLE_ANSWER;
  } else if (prefer === 'low') {
    preferCode = nn.PREFER_LOW_POWER;
  } else {
    preferCode = nn.PREFER_FAST_SINGLE_ANSWER;
  }
  return preferCode;
};

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
  return searchParams.has('m') ? searchParams.get('m') : 'none';
}

const getExampleDirectoryName = () => {
  const pathArray = location.pathname.split('/');
  const dirName = pathArray[pathArray.length - 2];
  return dirName.replace(/_/g, ' ');
};
