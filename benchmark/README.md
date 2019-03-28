WebML Benchmark
======
This benchmark loads:
-   Models trained by ImageNet
    - Mobilenet V1 (TFLite)
    - Mobilenet V1 Quant (TFLite)
    - Mobilenet V2 (TFLite)
    - Mobilenet V2 Quant (TFLite)
    - Squeezenet (TFLite)
    - Inception V3 (TFLite)
    - Inception V4 (TFLite)
    - Inception Resnet V2 (TFLite)
    - Mobilenet v2 (ONNX)
    - Squeezenet (ONNX)
    - Resnet-50 v1 (ONNX)
    - Resnet-50 v2 (ONNX)
    - Inception v2 (ONNX)
    - DenseNet (ONNX)
-   Object Detection Models
    - SSD MobileNet v1 (TFLite)
    - SSD MobileNet v1 Quant (TFLite)
    - SSD MobileNet v2 (TFLite)
    - SSD MobileNet v2 Quant (TFLite)
    - SSDLite MobileNet v2 (TFLite)
    - Tiny Yolo v2 COCO (TFLite)
    - Tiny Yolo v2 VOC (TFLite)
-   PoseNet Model using [Model: 1.0 / OutputStride: 16 / Scale Factor: 0.5 / Score Threshold: 0.5 parameters](../examples/skeleton_detection/README.md)
-   Semantic Segmentation Models
    - Deeplab 224 (TFLite)
    - Deeplab 224 Atrous (TFLite)
    - Deeplab 257 (TFLite)
    - Deeplab 257 Atrous (TFLite)
    - Deeplab 321 (TFLite)
    - Deeplab 321 Atrous (TFLite)
    - Deeplab 513 (TFLite)
    - Deeplab 513 Atrous (TFLite)
, constructs and inferences it by WebML API.

Perpare Model
-----------
Before run this benchmark with **Models trained by ImageNet**, you need to download those models. Please check out [README.md](../examples/image_classification/model/README.md) in ../examples/image_classification/model folder for details.

Before run this benchmark with **Object Detection Models**, please refer to [README.md](../examples/object_detection/model/README.md) in ../examples/object_detection/model folder for details.

Before run this benchmark with **Semantic Segmentation Models**, please refer to follow [README.md](../examples/semantic_segmentation/model/README.md) in ../examples/semantic_segmentation/model folder for details.
