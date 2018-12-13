# Prerequisites

Download the following models to this directory:

- [DeepLab (with Atrous Conv)](https://drive.google.com/file/d/1La9pi75J6RwSkgYcme1d9FLL4LZ3_JnE/view?usp=sharing)
- [DeepLab](https://drive.google.com/file/d/15l8kxoM0JBXv3Nd-BpyAQ7oZpgJWkKd3/view?usp=sharing)

This directory should contain two files

```txt
deeplab_mobilenetv2_513.tflite
deeplab_mobilenetv2_513_dilated.tflite
```

These two models are equivalent. The latter is used for platforms (Android) that do not natively support atrous convolution. They are converted from this [DeepLab checkpoint](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz). You can follow the steps below to convert your own model.

## Convert `.pb` to `.tflite`

1. Download source

We suggest you download the source since we need to take advantage of the [Graph Transform Tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md) later.

```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

2. Prune unused nodes

You can use either of the following two ways

```sh
# for tensorflow installed from package manager
python -m tensorflow.python.tools.optimize_for_inference \
--input=/path/to/frozen_inference_graph.pb \
--output=/path/to/frozen_inference_graph_stripped.pb \
--frozen_graph=True \
--input_names="sub_7" \
--output_names="ResizeBilinear_3"
```

```sh
# build from source
bazel build tensorflow/python/tools:optimize_for_inference
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input=/path/to/frozen_inference_graph.pb \
--output=/path/to/frozen_inference_graph_stripped.pb \
--frozen_graph=True \
--input_names="sub_7" \
--output_names="ResizeBilinear_3"
```

3. Flatten atrous convolution

Graph Transform Tool will assist you to substitute sequences of `SpaceToBatchND` - `(Depthwise)Conv2D` - `BatchToSpaceND` in grapgh with regular `(Depthwise)Conv2d`s in which kernels are padded with zeros.

```sh
# build from source
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph="/path/to/frozen_inference_graph_stripped.pb" \
--out_graph="/path/to/frozen_inference_graph_flatten.pb" \
--inputs='sub_7' \
--outputs='ResizeBilinear_3' \
--transforms='flatten_atrous_conv'
```

4. Export TFLite

You can use either of the following two ways

```sh
# for tensorflow installed from package manager
tflite_convert \
--graph_def_file=frozen_inference_graph_flatten.pb \
--output_file=deeplab_mobilenetv2_513.tflite \
--output_format=TFLITE \
--input_format=TENSORFLOW_GRAPHDEF \
--input_arrays=sub_7 \
--output_arrays=ResizeBilinear_3
```

```sh
# build from source
bazel build tensorflow/lite/toco:toco
bazel-bin/tensorflow/lite/toco/toco \
--input_file="/path/to/frozen_inference_graph_flatten.pb" \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--output_file="/path/to/deeplab_mobilenetv2_513.tflite" \
--inference_type=FLOAT \
--input_type=FLOAT \
--input_arrays=sub_7 \
--output_arrays=ResizeBilinear_3 \
--input_shapes=1,513,513,3
```
