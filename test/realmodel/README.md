# Generate realmodel testcase tool 

## Prerequisites

### 1. Follow [README](https://github.com/intel/webml-polyfill/blob/master/README.md), to launch HTTP Server for WebML.

### 2. Set config.json file

* `localURL`: url for local index.html.
* `backend`: backend value.
* `modelName`: model name.
* `url`: download model file url.

### 3. Support modelName and url.
* [Squeezenet](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz)
* [MobileNet v2-1.0](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz)
* [ResNet-50 v1](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.tar.gz)
* [ResNet-50 v2](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz)
* [Inception v2](https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz)(need rename inception_v2/model.onnx to inception_v2/inception_v2.onnx)

## Install
```sh
$ npm install
```

## Command

### Run all in one step
```sh
$ npm run all
```

### Download Model tar package
```sh
$ npm run downloadModel
```

### Unzip Model tar package
```sh
$ npm run unzipTar
```

### Get TestCase resources origin data
```sh
$ npm run getTCRes
```

### Generate TestCase
```sh
$ npm run genTC
```

### Generate all in one TestCase Html
```sh
$ npm run genHtml
```


## Output
* `../realmodel-<modelName>.html`: auto generated all in one TestCase Html.
* `./testcase`: auto generated folder for saving TestCase files.
* `./testcase/res`: auto generated folder for resource files.
* `./model`: auto generated folder for saving model files.
