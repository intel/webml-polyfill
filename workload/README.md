WebML Workload
======
This workload loads:
-   Image Classification Models
    - MobileNet v1 (TFLite)
    - MobileNet v1 Quant (TFLite)
    - MobileNet v2 (TFLite)
    - MobileNet v2 Quant (TFLite)
    - SqueezeNet (TFLite)
    - Inception v3 (TFLite)
    - Inception v3 Quant (TFLite)
    - Inception v4 (TFLite)
    - Inception v4 Quant (TFLite)
    - Inception ResNet v2 (TFLite)
    - SqueezeNet (ONNX)
    - MobileNet v2 (ONNX)
    - ResNet50 v1 (ONNX)
    - ResNet50 v2 (ONNX)
    - Inception v2 (ONNX)
    - DenseNet 121 (ONNX)
    - SqueezeNet (OpenVino)
    - MobileNet v1 (OpenVino)
    - MobileNet v2 (OpenVino)
    - ResNet50 v1 (OpenVino)
    - DenseNet 121 (OpenVino)
    - Inception v2 (OpenVino)
    - Inception v4 (OpenVino)
-   Object Detection Models
    - SSD MobileNet v1 (TFLite)
    - SSD MobileNet v1 Quant (TFLite)
    - SSD MobileNet v2 (TFLite)
    - SSD MobileNet v2 Quant (TFLite)
    - SSDLite MobileNet v2 (TFLite)
    - Tiny Yolo v2 COCO (TFLite)
    - Tiny Yolo v2 VOC (TFLite)
-   Skeleton Detection Models
    - PoseNet Model using [Model: 1.0 / OutputStride: 16 / Scale Factor: 0.5 / Score Threshold: 0.5 parameters](../examples/skeleton_detection/README.md)
-   Semantic Segmentation Models
    - Deeplab 224 (TFLite)
    - Deeplab 224 Atrous (TFLite)
    - Deeplab 257 (TFLite)
    - Deeplab 257 Atrous (TFLite)
    - Deeplab 321 (TFLite)
    - Deeplab 321 Atrous (TFLite)
    - Deeplab 513 (TFLite)
    - Deeplab 513 Atrous (TFLite)
-   Super Resolution Models
    - SRGAN 96x4 (TFLite)
    - SRGAN 128x4 (TFLite)
-   Facial Landmark Detection Models
    - SSD MobileNet v1 Face (TFLite)
    - SSD MobileNet v2 Face (TFLite)
    - SSDLite MobileNet v2 Face (TFLite)
    - Tiny Yolo v2 Face (TFLite)
    - Facial Landmark (TFLite)
-   Emotion Analysis Models
    - SSD MobileNet v1 Face (TFLite)
    - SSD MobileNet v2 Face (TFLite)
    - SSDLite MobileNet v2 Face (TFLite)
    - Tiny Yolo v2 Face (TFLite)
    - Emotion Classification (TFLite)

, constructs and inferences it by WebML API.

Perpare Model
-----------
Before run this workload with **Image Classification Models**, you need to download those models. Please check out [README.md](../examples/image_classification/model/README.md) in ../examples/image_classification/model folder for details.

Before run this workload with **Object Detection Models**, please refer to [README.md](../examples/object_detection/model/README.md) in ../examples/object_detection/model folder for details.

Before run this workload with **Skeleton Detection Models**, please refer to follow [README.md](../examples/skeleton_detection/model/README.md) in ../examples/skeleton_detection/model folder for details.

Before run this workload with **Semantic Segmentation Models**, please refer to follow [README.md](../examples/semantic_segmentation/model/README.md) in ../examples/semantic_segmentation/model folder for details.

Before run this workload with **Super Resolution Models**, please refer to follow [README.md](../examples/super_resolution/model/README.md) in ../examples/super_resolution/model folder for details.

Before run this workload with **Facial Landmark Detection Models**, please refer to follow [README.md](../examples/facial_landmark_detection/model/README.md) in ../examples/facial_landmark_detection/model folder for details.

Before run this workload with **Emotion Analysis Models**, please refer to follow [README.md](../examples/emotion_analysis/model/README.md) in ../examples/emotion_analysis/model folder for details.
