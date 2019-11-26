# Generate realmodel testcase tool 

## Prerequisites

### 1. Follow [README](https://github.com/intel/webml-polyfill/blob/master/README.md), to launch HTTP Server for WebML.

### 2. Set config.json file

* `localURL`: Local URL to browser test/realmodel/index.html
* `modelName`: Target ONNX model name 
* `url`: Download URL of target ONNX model package file


### 3. Supported ONNX.js models

|  ModelName |   URL   |
|  :---:  |  :---  |
|  squeezenet1.1  |   https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz  | 
|  mobilenetv2-1.0  |   https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz  |
|  resnet50v1  |   https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.tar.gz  | 
|  resnet50v2  |   https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz  |

## Install
```sh
$ npm install
```

## Command

### Run all-in-one step(this is equivalent to the following five steps)
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

### Generate all-in-one TestCase Html
```sh
$ npm run genHtml
```


## Output
* `../<modelName>.html`: auto generated all-in-one TestCase Html.
* `./testcase/<modelName>`: auto generated folder for saving TestCase files.
* `./testcase/res`: auto generated folder for resource files.
* `./model`: auto generated folder for saving model files.



### Finally
* for example (`open chromium and visit localhost:8080/test/<modelName>.html`)
