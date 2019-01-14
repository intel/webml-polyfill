WebML Benchmark
======
This benchmark loads:
-   Models trained by ImageNet
    - Mobilenet V1(TFlite)
    - Mobilenet V2(TFlite)
    - Inception V3(TFlite)
    - Inception V4(TFlite)
    - Squeezenet(TFlite)
    - Inception Resnet V2(TFlite)
    - Squeezenet(Onnx)
    - Mobilenet v2(Onnx)
    - Resnet-50 v1(Onnx)
    - Resnet-50 v2(Onnx)
    - Inception v2(Onnx)
    - DenseNet(Onnx)
-   SSD MobileNet Model trained by COCO in TensorFlow Lite format
-   PoseNet Model using [Model: 1.0 / OutputStride: 16 / Scale Factor: 0.5 / Score Threshold: 0.5 parameters](../examples/posenet/README.md)

, constructs and inferences it by WebML API.

Download Model
-----------
Before run this benchmark with **Models trained by ImageNet**, you need to download those models. Please check out [README.md](../examples/image_classification/model/README.md) in ../examples/image_classification/model folder for details.

Before run this benchmark with **SSD MobileNet Model**, you need to download the model. Please check out [README.md](../examples/ssd_mobilenet/model/README.md) in ../examples/ssd_mobilenet/model folder for details.