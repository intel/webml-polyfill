Download `.onnx` models into current directory

1. [Squeezenet](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx)
2. [Mobilenet v2](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx)
3. [Resnet-18 v1](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx)
4. [Resnet-18 v2](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx)
5. [Inception v2](https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz) (Untar it and rename `model.onnx` to `inceptionv2.onnx`)

This directory should contain 5 model files

```
squeezenet1.1.onnx
mobilenetv2-1.0.onnx
resnet18v1.onnx
resnet18v2.onnx
inceptionv2.onnx
```

And we have provided the labels files:
```
labels.txt
ilsvrc2012labels.txt
```

Check out [ONNX models](https://github.com/onnx/models) for more details.