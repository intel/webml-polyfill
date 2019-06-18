## WebNN API Image Classification Example
This example loads Image classification models trained by ImageNet, constructs and inferences it by WebNN API.

### Download Model
Before launch this example, you need to download the model. Please check out [README.md](model/README.md) in model folder for details.

### Screenshot
![screenshot](screenshot.png)

### URL Parameters for Image Classification Example
E.g. 
https://127.0.0.1/examples/image_classification/index.html?prefer=none&b=WASM&m=mobilenet_v1_tflite&s=image&d=0

#### Description
| Parameter | Value | Description | Note |
|----|------|------|-----------|
| prefer | sustained, fast, low | Preferred backend for WebNN API backend<br>sustained == GPU<br>fast == CPU<br>low == Low Power |Only work for WebNN API backend, useless when backend is WASM or WebGL |
| b | WASM, WebGL, WebML | Backend | Case sensitive |
| m | mobilenet_v1_tflite, squeezenet_onnx, mobilenet_v2_openvino, etc.s | Unique ID for model | Align with `modelId` defined in `../util/base.js`|
| s | image <br>camera | Show image or camera tab directly | |
| d | 0, 1  | Display model<br>// full view <br>0<br>// compact view<br>1  | |


### Netron URLs
It supports to use Netron Visualizer for deep learning and machine learning models in [model.html](../model.html), please upload [these models](model/README.md) to your website (or CDN/OSS services).