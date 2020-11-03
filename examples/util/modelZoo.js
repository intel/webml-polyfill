const modelZoo = {
  imageClassificationModels: [{
    modelName: 'MobileNet v1 (TFLite)',
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN', 'OpenCV.js'],
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
    framework: ['WebNN', 'OpenCV.js'],
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
    framework: ['WebNN', 'OpenCV.js'],
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
    framework: ['WebNN', 'OpenCV.js'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'squeezenet_openvino',
    modelSize: '4.9MB',
    modelFile: '../image_classification/model/squeezenet1.1.bin',
    labelsFile: '../image_classification/model/labels1000.txt',
    inputSize: [227, 227, 3],
    outputSize: 1000,
    preOptions: {
      channelScheme: 'BGR',
      nchwFlag: true,
    },
    intro: 'A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters.',
    paperUrl: 'https://arxiv.org/abs/1602.07360'
  }, {
    modelName: 'MobileNet v1 (OpenVino)',
    framework: ['WebNN'],
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
      nchwFlag: true,
    },
    intro: 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
    paperUrl: 'https://arxiv.org/pdf/1704.04861.pdf'
  }, {
    modelName: 'MobileNet v1 Quant (OpenVino)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'mobilenet_v1_quant_openvino',
    isQuantized: true,
    isIE: true,
    modelSize: '4.4MB',
    modelFile: '../image_classification/model/mobilenet_v1_1.0_224_quant.bin',
    labelsFile: '../image_classification/model/labels1001.txt',
    inputSize: [224, 224, 3],
    outputSize: 1001,
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      nchwFlag: true,
    },
    intro: 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
    paperUrl: 'https://arxiv.org/pdf/1704.04861.pdf'
  }, {
    modelName: 'MobileNet v2 (OpenVino)',
    framework: ['WebNN'],
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
      nchwFlag: true,
    },
    intro: 'MobileNetV2 improves the state of the art performance of mobile models.',
    paperUrl: 'https://arxiv.org/abs/1801.04381'
  }, {
    modelName: 'MobileNet v2 Quant (OpenVino)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'mobilenet_v2_quant_openvino',
    isQuantized: true,
    isIE: true,
    modelSize: '3.7MB',
    modelFile: '../image_classification/model/mobilenet_v2_1.0_224_quant.bin',
    labelsFile: '../image_classification/model/labels1001.txt',
    inputSize: [224, 224, 3],
    outputSize: 1001,
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      nchwFlag: true,
    },
    intro: 'MobileNetV2 improves the state of the art performance of mobile models.',
    paperUrl: 'https://arxiv.org/abs/1801.04381'
  }, {
    modelName: 'ResNet50 v1 (OpenVino)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'resnet50_v1_openvino',
    modelSize: '102.1MB',
    modelFile: '../image_classification/model/resnet-50.bin',
    labelsFile: '../image_classification/model/labels1001.txt',
    inputSize: [224, 224, 3],
    outputSize: 1001,
    preOptions: {
      mean: [123.68,116.78,103.94],
      nchwFlag: true,
    },
    intro: 'A residual learning framework to ease the training of networks that are substantially deeper than those used previously. This result won the 1st place on the ILSVRC 2015 classification task.',
    paperUrl: 'https://arxiv.org/abs/1512.03385'
  }, {
    modelName: 'ResNet50 v1 Quant (OpenVino)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'resnet50_v1_quant_openvino',
    isQuantized: true,
    isIE: true,
    modelSize: '25.8MB',
    modelFile: '../image_classification/model/resnet-50_quant.bin',
    labelsFile: '../image_classification/model/labels1001.txt',
    inputSize: [224, 224, 3],
    outputSize: 1001,
    preOptions: {
      mean: [123.68,116.78,103.94],
      nchwFlag: true,
    },
    intro: 'A residual learning framework to ease the training of networks that are substantially deeper than those used previously. This result won the 1st place on the ILSVRC 2015 classification task.',
    paperUrl: 'https://arxiv.org/abs/1512.03385'
  }, {
    modelName: 'DenseNet 121 (OpenVino)',
    framework: ['WebNN'],
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
      nchwFlag: true,
    },
    postOptions: {
      softmax: true,
    },
    intro: 'Dense Convolutional Network (DenseNet) connects each layer to every other layer in a feed-forward fashion. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. ',
    paperUrl: 'https://arxiv.org/abs/1608.06993'
  }, {
    modelName: 'Inception v2 (OpenVino)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'inception_v2_openvino',
    modelSize: '44.7MB',
    modelFile: '../image_classification/model/googlenet-v2.bin',
    labelsFile: '../image_classification/model/ilsvrc2012labels.txt',
    inputSize: [224, 224, 3],
    outputSize: 1000,
    preOptions: {
      channelScheme: 'BGR',
      nchwFlag: true,
    },
    intro: 'Inception-v2 is trained for the ImageNet Large Visual Recognition Challenge.',
    paperUrl: 'https://arxiv.org/abs/1512.00567'
  }, {
    modelName: 'Inception v4 (OpenVino)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'inception_v4_openvino',
    modelSize: '170.6MB',
    modelFile: '../image_classification/model/googlenet-v4.bin',
    labelsFile: '../image_classification/model/labels1001.txt',
    inputSize: [299, 299, 3],
    outputSize: 1001,
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      nchwFlag: true,
    },
    intro: 'Inception architecture that has been shown to achieve very good performance at relatively low computational cost.',
    paperUrl: 'https://arxiv.org/abs/1602.07261'
  }, {
    modelName: 'Squeezenet (Caffe2)',
    framework: ['WebNN'],
    format: 'Caffe2',
    modelId: 'Squeezenet_Caffe2',
    modelSize: '6.2MB',
    inputSize: [227, 227, 3],
    outputSize: 1000,
    modelFile: '../image_classification/model/squeezenet_predict.pb',
    labelsFile: '../image_classification/model/labels1000.txt',
    preOptions: {
      mean: [104, 117, 123],
      std: [1, 1, 1],
      channelScheme: 'BGR',
    },
    intro: 'A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters.',
    paperUrl: 'https://arxiv.org/abs/1602.07360'
  }, {
    modelName: 'MobileNet v1 Quant (Caffe2)',
    framework: ['WebNN'],
    format: 'Caffe2',
    modelId: 'mobilenet_v1_quant_caffe2',
    modelSize: '4.5MB',
    isQuantized: true,
    isDNNL: true,
    inputSize: [224, 224, 3],
    outputSize: 1001,
    modelFile: '../image_classification/model/mobilenet_v1_quant_predict.pb',
    labelsFile: '../image_classification/model/labels1001.txt',
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [1, 1, 1],
      rescaleSize: 256,
    },
    intro: 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
    paperUrl: 'https://arxiv.org/pdf/1704.04861.pdf'
  }],

  objectDetectionModels: [{
    modelName: 'SSD MobileNet v1 (TFLite)',
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'ssd_mobilenet_v1_tflite',
    modelSize: '27.3MB',
    modelFile: '../object_detection/model/ssd_mobilenet_v1.tflite',
    labelsFile: '../object_detection/model/coco_classes.txt',
    category: 'SSD',
    boxSize: 4,
    numClasses: 91,
    numBoxes: [1083, 600, 150, 54, 24, 6],
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
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'ssd_mobilenet_v1_quant_tflite',
    isQuantized: true,
    modelSize: '6.9MB',
    modelFile: '../object_detection/model/ssd_mobilenet_v1_quant.tflite',
    labelsFile: '../object_detection/model/coco_classes.txt',
    category: 'SSD',
    boxSize: 4,
    numClasses: 91,
    numBoxes: [1083, 600, 150, 54, 24, 6],
    margin: [1, 1, 1, 1],
    inputSize: [300, 300, 3],
    intro: 'Quantized version of SSD Mobilenet v1',
    paperUrl: 'https://arxiv.org/pdf/1712.05877.pdf'
  }, {
    modelName: 'SSD MobileNet v2 (TFLite)',
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'ssd_mobilenet_v2_tflite',
    modelSize: '67.3MB',
    modelFile: '../object_detection/model/ssd_mobilenet_v2.tflite',
    labelsFile: '../object_detection/model/coco_classes.txt',
    category: 'SSD',
    boxSize: 4,
    numClasses: 91,
    numBoxes: [1083, 600, 150, 54, 24, 6],
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
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'ssd_mobilenet_v2_quant_tflite',
    isQuantized: true,
    modelSize: '6.2MB',
    modelFile: '../object_detection/model/ssd_mobilenet_v2_quant.tflite',
    labelsFile: '../object_detection/model/coco_classes.txt',
    category: 'SSD',
    boxSize: 4,
    numClasses: 91,
    numBoxes: [1083, 600, 150, 54, 24, 6],
    margin: [1, 1, 1, 1],
    inputSize: [300, 300, 3],
    intro: 'Quantized version of SSD Mobilenet v2',
    paperUrl: 'https://arxiv.org/abs/1806.08342'
  }, {
    modelName: 'SSDLite MobileNet v2 (TFLite)',
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'ssdlite_mobilenet_v2_tflite',
    modelSize: '17.9MB',
    modelFile: '../object_detection/model/ssdlite_mobilenet_v2.tflite',
    labelsFile: '../object_detection/model/coco_classes.txt',
    category: 'SSD',
    boxSize: 4,
    numClasses: 91,
    numBoxes: [1083, 600, 150, 54, 24, 6],
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
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'tiny_yolov2_coco_tflite',
    modelSize: '44.9MB',
    modelFile: '../object_detection/model/tiny_yolov2_coco.tflite',
    labelsFile: '../object_detection/model/coco_classes_part.txt',
    category: 'YOLO',
    numClasses: 80,
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
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'tiny_yolov2_voc_tflite',
    modelSize: '63.4MB',
    modelFile: '../object_detection/model/tiny_yolov2_voc.tflite',
    labelsFile: '../object_detection/model/pascal_classes.txt',
    category: 'YOLO',
    numClasses: 20,
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
    framework: ['WebNN'],
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
    modelName: 'Deeplab_224 (TFLite)',
    framework: ['WebNN'],
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
    modelName: 'Deeplab_224_Atrous (TFLite)',
    framework: ['WebNN'],
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
    modelName: 'Deeplab_257 (TFLite)',
    framework: ['WebNN'],
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
    modelName: 'Deeplab_257_Atrous (TFLite)',
    framework: ['WebNN'],
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
    modelName: 'Deeplab_321 (TFLite)',
    framework: ['WebNN'],
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
    modelName: 'Deeplab_321_Atrous (TFLite)',
    framework: ['WebNN'],
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
    modelName: 'Deeplab_513 (TFLite)',
    framework: ['WebNN'],
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
    modelName: 'Deeplab_513_Atrous (TFLite)',
    framework: ['WebNN'],
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
  }, {
    modelName: 'Deeplab_224_Atrous (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'deeplab_mobilenet_v2_224_atrous_openvino',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_224_dilated.bin',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [224, 224, 3],
    outputSize: [224, 224, 1],
    preOptions: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      nchwFlag: true,
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab_224_Quant (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'deeplab_224_quant_openvino',
    isQuantized: true,
    isIE: true,
    modelSize: '2.3MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_224_dilated_quant.bin',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [224, 224, 3],
    outputSize: [224, 224, 1],
    preOptions: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      nchwFlag: true,
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab_257_Atrous (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'deeplab_mobilenet_v2_257_atrous_openvino',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_257_dilated.bin',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [257, 257, 3],
    outputSize: [257, 257, 1],
    preOptions: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      nchwFlag: true,
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab_257_Quant (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'deeplab_257_quant_openvino',
    isQuantized: true,
    isIE: true,
    modelSize: '2.3MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_257_dilated_quant.bin',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [257, 257, 3],
    outputSize: [257, 257, 1],
    preOptions: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      nchwFlag: true,
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab_321_Atrous (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'deeplab_mobilenet_v2_321_atrous_openvino',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_321_dilated.bin',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [321, 321, 3],
    outputSize: [321, 321, 1],
    preOptions: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      nchwFlag: true,
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab_321_Quant (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'deeplab_321_quant_openvino',
    isQuantized: true,
    isIE: true,
    modelSize: '2.3MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_321_dilated_quant.bin',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [321, 321, 3],
    outputSize: [321, 321, 1],
    preOptions: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      nchwFlag: true,
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab_513_Atrous (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'deeplab_mobilenet_v2_513_atrous_openvino',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_513_dilated.bin',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 1],
    preOptions: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      nchwFlag: true,
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab_513_Quant (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'deeplab_513_quant_openvino',
    isQuantized: true,
    isIE: true,
    modelSize: '2.3MB',
    modelFile: '../semantic_segmentation/model/deeplab_mobilenetv2_513_dilated_quant.bin',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 1],
    preOptions: {
      mean: [0, 0, 0],
      std: [1, 1, 1],
      nchwFlag: true,
    },
    intro: 'Equivalent to the model above (without dilated suffix) but only available on platforms that natively support atrous convolution.',
    paperUrl: 'https://arxiv.org/abs/1802.02611'
  }, {
    modelName: 'Deeplab_224 (Tensorflow)',
    framework: ['OpenCV.js'],
    format: 'Tensorflow',
    modelId: 'opt_deeplabv3_mnv2_224',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/opt_deeplabv3_mnv2_224.pb',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [224, 224, 3],
    outputSize: [224, 224, 1],
    preOptions: {
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      norm: true
    },
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1606.00915'
  }, {
    modelName: 'Deeplab_257 (Tensorflow)',
    framework: ['OpenCV.js'],
    format: 'Tensorflow',
    modelId: 'opt_deeplabv3_mnv2_257',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/opt_deeplabv3_mnv2_257.pb',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [257, 257, 3],
    outputSize: [257, 257, 1],
    preOptions: {
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      norm: true
    },
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1606.00915'
  }, {
    modelName: 'Deeplab_321 (Tensorflow)',
    framework: ['OpenCV.js'],
    format: 'Tensorflow',
    modelId: 'opt_deeplabv3_mnv2_321',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/opt_deeplabv3_mnv2_321.pb',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [321, 321, 3],
    outputSize: [321, 321, 1],
    preOptions: {
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      norm: true
    },
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1606.00915'
  }, {
    modelName: 'Deeplab_513 (Tensorflow)',
    framework: ['OpenCV.js'],
    format: 'Tensorflow',
    modelId: 'opt_deeplabv3_mnv2_513',
    modelSize: '8.4MB',
    modelFile: '../semantic_segmentation/model/opt_deeplabv3_mnv2_513.pb',
    labelsFile: '../semantic_segmentation/model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 1],
    preOptions: {
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      norm: true
    },
    intro: 'DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels to every pixel in the input image.',
    paperUrl: 'https://arxiv.org/abs/1606.00915'
  }],

  superResolutionModels: [
    {
      modelName: 'SRGAN 96x4 (TFLite)',
      framework: ['WebNN'],
      format: 'TFLite',
      modelId: 'srgan_96x4_tflite',
      modelSize: '6.1MB',
      inputSize: [96, 96, 3],
      outputSize: [384, 384, 3],
      scale: 4,
      modelFile: '../super_resolution/model/srgan_96_4.tflite',
      preOptions: {
        mean: [127.5, 127.5, 127.5],
        std: [127.5, 127.5, 127.5],
      },
      intro: 'Photo-realistic single image Super-Resolution using a generative adversarial network.',
      paperUrl: 'https://arxiv.org/abs/1609.04802'
    },
    {
      modelName: 'SRGAN 128x4 (TFLite)',
      framework: ['WebNN'],
      format: 'TFLite',
      modelId: 'srgan_128x4_tflite',
      modelSize: '6.1MB',
      inputSize: [128, 128, 3],
      outputSize: [512, 512, 3],
      scale: 4,
      modelFile: '../super_resolution/model/srgan_128_4.tflite',
      preOptions: {
        mean: [127.5, 127.5, 127.5],
        std: [127.5, 127.5, 127.5],
      },
      intro: 'Photo-realistic single image Super-Resolution using a generative adversarial network.',
      paperUrl: 'https://arxiv.org/abs/1609.04802'
    }
  ],

  faceDetectionModels: [{
    modelName: 'SSD MobileNet v1 (TFlite)',
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'ssd_mobilenetv1_face_tflite',
    modelSize: '22.0MB',
    category: 'SSD',
    modelFile: '../facial_landmark_detection/model/ssd_mobilenetv1_face.tflite',
    boxSize: 4,
    numClasses: 2,
    numBoxes: [1083, 600, 150, 54, 24, 6],
    margin: [1.2, 1.2, 0.8, 1.1],
    inputSize: [300, 300, 3],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'This SSD Mobilenet V1 is based on SSD Mobilenet V1 model structure, and is trained by Tensorflow Object Detection API with WIDER_FACE dataset for face detection task.',
    paperUrl: 'https://arxiv.org/abs/1803.08225'
  }, {
    modelName: 'SSD MobileNet v2 (TFlite)',
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'ssd_mobilenetv2_face_tflite',
    modelSize: '18.4MB',
    category: 'SSD',
    modelFile: '../facial_landmark_detection/model/ssd_mobilenetv2_face.tflite',
    boxSize: 4,
    numClasses: 2,
    numBoxes: [1083, 600, 150, 54, 24, 6],
    margin: [1.2, 1.2, 0.8, 1.1],
    inputSize: [300, 300, 3],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'This SSD Mobilenet V2 is based on SSD Mobilenet V2 model structure, and is trained by Tensorflow Object Detection API with WIDER_FACE dataset for face detection task.',
    paperUrl: 'https://arxiv.org/abs/1801.04381'
  }, {
    modelName: 'SSDLite MobileNet v2 (TFlite)',
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'ssdlite_mobilenetv2_face_tflite',
    modelSize: '12.1MB',
    category: 'SSD',
    modelFile: '../facial_landmark_detection/model/ssdlite_mobilenetv2_face.tflite',
    boxSize: 4,
    numClasses: 2,
    numBoxes: [1083, 600, 150, 54, 24, 6],
    margin: [1.2, 1.2, 0.8, 1.1],
    inputSize: [300, 300, 3],
    preOptions: {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
    },
    intro: 'This SSDLite Mobilenet V2 is based on SSDLite Mobilenet V2 model structure, and is trained by Tensorflow Object Detection API with WIDER_FACE dataset for face detection task.',
    paperUrl: 'https://arxiv.org/abs/1801.04381'
  }, {
    modelName: 'Tiny Yolo v2 Face (TFlite)',
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'tiny_yolov2_face_tflite',
    modelSize: '44.1MB',
    modelFile: '../facial_landmark_detection/model/tiny_yolov2_face.tflite',
    category: 'YOLO',
    margin: [1.15, 1.15, 0.6, 1.15],
    inputSize: [416, 416, 3],
    outputSize: 1 * 13 * 13 * 30,
    anchors: [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
    preOptions: {
      norm: true,
    },
    intro: 'This Tiny YOLO V2 is based off the Darknet reference network and trained with WIDER_FACE dataset for face detection task.',
    paperUrl: 'https://arxiv.org/abs/1612.08242'
  }],

  facialLandmarkDetectionModels: [{
    modelName: 'SimpleCNN (TFlite)',
    framework: ['WebNN'],
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
    modelName: 'Simple CNN (TFlite)',
    framework: ['WebNN'],
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
    framework: ['WebNN'],
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
  }, {
    modelName: 'KWS DNN (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'kws_dnn_openvino',
    modelSize: '320kB',
    inputSize: [1, 250],
    outputSize: [1, 12],
    sampleRate: 16000,
    modelFile: '../speech_commands/model/kws_dnn.bin',
    labelsFile: '../speech_commands/model/labels2.txt',
    mfccsOptions: {
      windowSize: 640,
      windowStride: 640,
      upperFrequencyLimit: 4000,
      lowerFrequencyLimit: 20,
      filterbankChannelCount: 40,
      dctCoefficientCount: 10,
    },
    preOptions: {
      nchwFlag: true,
    },
    intro: 'A small-footprint keyword spotting (KWS) on Microcontroller.',
    paperUrl: 'https://arxiv.org/pdf/1711.07128.pdf'
  }],

  faceRecognitionModels: [{
    modelName: 'FaceNet (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVino',
    modelId: 'facenet_recognition_openvino',
    modelSize: '93.9MB',
    modelFile: '../face_recognition/model/facenet.bin',
    inputSize: [160, 160, 3],
    outputSize: 512,
    preOptions: {
      mean: [0, 0, 0, 0],
      std: [1, 1, 1, 1],
      channelScheme: 'BGR',
      nchwFlag: true,
    },
    postOptions: {
      distanceMetric: 'euclidean',
      threshold: 1.26
    },
    intro: 'This is a CNN model for face recognition which learns discriminative features of faces and produces embeddings for input face images.',
    paperUrl: 'https://arxiv.org/abs/1503.03832'
  }, {
    modelName: 'Face Reidentification (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVino',
    modelId: 'facereidentification_recognition_openvino',
    modelSize: '4.4MB',
    modelFile: '../face_recognition/model/face-reidentification-retail-0095.bin',
    inputSize: [128, 128, 3],
    outputSize: 256,
    preOptions: {
      mean: [0, 0, 0, 0],
      std: [1, 1, 1, 1],
      channelScheme: 'BGR',
      nchwFlag: true,
    },
    postOptions: {
      distanceMetric: 'cosine',
      threshold: 0.8
    },
    intro: 'This is a lightweight network for the face re-identification scenario that is\
      based on MobileNet V2. The model produces feature vectors which should be close in cosine distance for similar faces and far for different faces.'
  }],

  speechRecognitionModels: [{
    modelName: 'wsj_dnn5b (OpenVINO)',
    framework: ['WebNN'],
    format: 'OpenVINO',
    modelId: 'wsj_dnn5b_openvino',
    modelSize: '115.6MB',
    inputSize: [1, 440],
    outputSize: [1, 3425],
    modelFile: '../speech_recognition/model/wsj_dnn5b.bin',
    arkFile: '../speech_recognition/ark/dev93.ark',
    scoreFile: '../speech_recognition/ark/dev93_scores.ark',
     preOptions: {
      nchwFlag: true,
    },
    intro: 'This model was trained by the Kaldi s5 recipe and the Kaldi Nnet (nnet1) framework and using the Wall Street Journal corpus.',
    paperUrl: 'https://docs.openvinotoolkit.org/latest/_inference_engine_samples_speech_sample_README.html'
  }],

  styleTransferModels: [{
    modelName: 'Fast Style Transfer (TFLite)',
    framework: ['WebNN'],
    format: 'TFLite',
    modelId: 'fast_style_transfer_tflite',
    modelSize: '150.7kB',
    inputSize: [540, 540, 3],
    outputSize: [540, 540, 3],
    modelFile: '../style_transfer/model/starry-night.tflite',
    intro: 'Neural style transfer is the process of taking the style of one image then applying it to the content of another image.',
    paperUrl: 'https://arxiv.org/abs/1705.04058'
  }, {
    modelName: 'Fast Style Transfer (ONNX)',
    framework: ['WebNN'],
    format: 'ONNX',
    modelId: 'fast_style_transfer_onnx',
    modelSize: '150.7kB',
    inputSize: [540, 540, 3],
    outputSize: [540, 540, 3],
    modelFile: '../style_transfer/model/starry-night.onnx',
    intro: 'Neural style transfer is the process of taking the style of one image then applying it to the content of another image.',
    paperUrl: 'https://arxiv.org/abs/1705.04058'
  }]
};


// mobileNetArchitecture = [layerName, stride]
// conv2d: convolution layer
// separableConv: depthwise convolution layer + pointwise convolution layer
const mobileNet100Architecture = [
  ['conv2d', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1]
];

const mobileNet75Architecture = [
  ['conv2d', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1]
];

const mobileNet50Architecture = [
  ['conv2d', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1]
];

// ModelArch is for skeleton detection example
const ModelArch = new Map([
  [0.5, mobileNet50Architecture],
  [0.75, mobileNet75Architecture],
  [1.0, mobileNet100Architecture],
  [1.01, mobileNet100Architecture],
]);

// extract model lists into the browser env for backward compatiblity
for (const modelListName in modelZoo) {
  window[modelListName] = modelZoo[modelListName];
}

// model list maps the value in [key, value] of modelZoom
const getModelById = (modelList, id) => {
  for (const model of Object.values(modelList).flat()) {
    if (id === model.modelId) {
      return model;
    }
  }
  return null;
};

// models likes
//   {model: imageClassificationModels}
// or
//   {emotionAnalysis: emotionAnalysisModels, faceDetection: faceDetectionModels}
const getModelFromGivenModels = (modelsDict, id) => {
  for (const modelsList of Object.values(modelsDict)) {
    for (const model of modelsList) {
      if (id === model.modelId) {
        return model;
      }
    }
  }
  return null;
};

// current this function works for {model: xxxModels}
const selectModelFromGivenModels = (modelsDict, framework) => {
  let selectedDict = new Object();
  for(key of Object.keys(modelsDict)) {
    let newModelList = [];
    for (const model of modelsDict[key]) {
      if (model.framework && model.framework.includes(framework)) {
        newModelList.push(model);
      }
    }
    selectedDict[key] = newModelList;
  }
  
  return selectedDict;
};

const getFrameworkList = (modelsDict) => {
  let frameworkList = ['WebNN'];
  for (const modelsList of Object.values(modelsDict)) {
    for (const model of modelsList) {
      if (model.framework) {
        for (const frameworkItem of model.framework) {
          if (!frameworkList.includes(frameworkItem)) {
            frameworkList.push(frameworkItem);
          }
        }
      }
    }
  }
  return frameworkList;
};
