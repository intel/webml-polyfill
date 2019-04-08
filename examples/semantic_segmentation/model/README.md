# Prerequisites

Download all [DeepLab models](https://drive.google.com/open?id=1cMhKkGFc3DhJJWCWSPMGPLmfYZ3pVESW) to this directory. It should contain the following files:

```txt
deeplab_mobilenetv2_513_dilated.tflite
deeplab_mobilenetv2_513.tflite
deeplab_mobilenetv2_321_dilated.tflite
deeplab_mobilenetv2_321.tflite
deeplab_mobilenetv2_257_dilated.tflite
deeplab_mobilenetv2_257.tflite
deeplab_mobilenetv2_224_dilated.tflite
deeplab_mobilenetv2_224.tflite
```

They are all converted from this [frozen graph](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz). You can also follow the steps below to convert your own model.

Models without `dilated` suffix is used for platforms that do not natively support atrous convolution. They are equivalent to the suffixed versions in terms of the trained parameters.

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

## Export custom models

The default size of DeepLab is 513, which would result in considerably slow performance. You can export a model with a smaller size.

1. Download Tensorflow official models
```sh
git clone https://github.com/tensorflow/models.git
cd models/research/deeplab/
vim local_test_mobilenetv2.sh
```

2. Modify export parameters

For example, export a model with 321x321 inputs and outputs.

```sh
@@ -124,8 +124,8 @@ python "${WORK_DIR}"/export_model.py \
   --export_path="${EXPORT_PATH}" \
   --model_variant="mobilenet_v2" \
   --num_classes=21 \
-  --crop_size=513 \
-  --crop_size=513 \
+  --crop_size=321 \
+  --crop_size=321 \
```

3. Export model

```sh
./local_test_mobilenetv2.sh
```

The frozen graph is exported in `datasets/pascal_voc_seg/exp/train_on_trainval_set_mobilenetv2/export/frozen_inference_graph.pb`. You can then convert it to a TFLite Model per instructions above.
