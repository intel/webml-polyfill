WebML Benchmark
======
This benchmark loads:
  * MobileNet model trained by ImageNet in TensorFlow Lite format
  * SqueezeNet model trained by ImageNet in ONNX format
  * PoseNet model using [Model: 1.0 / OutputStride: 16 / Scale Factor: 0.5 / Score Threshold: 0.5 parameters](../examples/posenet/README.md)

, constructs and inferences it by WebML API.

Download Model
-----------
Before run this benchmark with MobileNet model, you need to download the model. Please check out [README.md](../examples/mobilenet/model/README.md) in ../examples/mobilenet/model folder for details.

Before run this benchmark with SqueezeNet model, you need to download the model. Please check out [README.md](../examples/squeezenet/model/README.md) in ../examples/squeezenet/model folder for details.
