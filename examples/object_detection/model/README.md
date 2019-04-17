## How to Download

Download [ssd_mobilenet_v1](https://drive.google.com/file/d/1JlAXwCQztZ-ySmIQ8rZtJ0pncaPhCDm5/view?usp=sharing), [ssd_mobilenet_v1_quant](https://drive.google.com/file/d/1NfvbEokyb2zTzICnqxNhXvJq_Av2BTUa/view?usp=sharing), [ssd_mobilenet_v2](https://drive.google.com/file/d/1JTotD3hmFL9ObHc-q-PlhBxYe3IqlQXH/view?usp=sharing), [ssd_mobilenet_v2_quant](https://drive.google.com/file/d/122FSeXPmcL-yt-0TAHPdjWYDV4Lme0oQ/view?usp=sharing), [ssdlite_mobilenet_v2](https://drive.google.com/file/d/1YWDKpyUnMG6L4ddmt4wGGvOjx17e-9Fg/view?usp=sharing), [tiny_yolov2_coco](https://drive.google.com/file/d/15xySV60owfc5tc8Z8IxRIELC4rT_16wt/view?usp=sharing), [tiny_yolov2_voc](https://drive.google.com/file/d/1fXksVZeVYsRyf_UnLDJf8-nkbCQmhW9J/view?usp=sharing), and put them here.

The model files are:

```txt
ssd_mobilenet_v1.tflite
ssd_mobilenet_v1_quant.tflite
ssd_mobilenet_v2.tflite
ssd_mobilenet_v2_quant.tflite
ssdlite_mobilenet_v2.tflite
tiny_yolov2_coco.tflite
tiny_yolov2_voc.tflite
```

## How to Generate

### For SSD Mobilenet Models

Check out [TensorFlow Lite Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for details.

These models are converted from [Tensorflow SSD MobileNet V1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz), [Tensorflow SSD MobileNet V1 Quant](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz), [Tensorflow SSD MobileNet V2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz), [Tensorflow SSD MobileNet V2 Quant](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz) and [Tensorflow SSDLite MobileNet V2](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz). You can use the following commands to convert your own model.

#### Float Models

```sh
python -m tensorflow.python.tools.optimize_for_inference \
--input=${download_model_dir}/frozen_inference_graph.pb \
--output=${out_dir}/frozen_inference_graph_stripped.pb --frozen_graph=True \
--input_names=Preprocessor/sub \
--output_names=\
"BoxPredictor_0/BoxEncodingPredictor/BiasAdd,BoxPredictor_0/ClassPredictor/BiasAdd,\
BoxPredictor_1/BoxEncodingPredictor/BiasAdd,BoxPredictor_1/ClassPredictor/BiasAdd,\
BoxPredictor_2/BoxEncodingPredictor/BiasAdd,BoxPredictor_2/ClassPredictor/BiasAdd,\
BoxPredictor_3/BoxEncodingPredictor/BiasAdd,BoxPredictor_3/ClassPredictor/BiasAdd,\
BoxPredictor_4/BoxEncodingPredictor/BiasAdd,BoxPredictor_4/ClassPredictor/BiasAdd,\
BoxPredictor_5/BoxEncodingPredictor/BiasAdd,BoxPredictor_5/ClassPredictor/BiasAdd" \
--alsologtostderr

tflite_convert \
--graph_def_file=${out_dir}/frozen_inference_graph_stripped.pb \
--output_file=${out_dir}/${model_name}.tflite \
--input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
--input_shapes=1,300,300,3 --input_arrays=Preprocessor/sub \
--output_arrays=\
"BoxPredictor_0/BoxEncodingPredictor/BiasAdd,BoxPredictor_0/ClassPredictor/BiasAdd,\
BoxPredictor_1/BoxEncodingPredictor/BiasAdd,BoxPredictor_1/ClassPredictor/BiasAdd,\
BoxPredictor_2/BoxEncodingPredictor/BiasAdd,BoxPredictor_2/ClassPredictor/BiasAdd,\
BoxPredictor_3/BoxEncodingPredictor/BiasAdd,BoxPredictor_3/ClassPredictor/BiasAdd,\
BoxPredictor_4/BoxEncodingPredictor/BiasAdd,BoxPredictor_4/ClassPredictor/BiasAdd,\
BoxPredictor_5/BoxEncodingPredictor/BiasAdd,BoxPredictor_5/ClassPredictor/BiasAdd" \
--inference_type=FLOAT --logtostderr
```

#### Quantize Models

```sh
tflite_convert \
--graph_def_file=${download_model_dir}/tflite_graph.pb --output_file=${out_dir}/${model_name}.tflite \
--input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays=\
"BoxPredictor_0/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,BoxPredictor_0/ClassPredictor/act_quant/FakeQuantWithMinMaxVars,\
BoxPredictor_1/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,BoxPredictor_1/ClassPredictor/act_quant/FakeQuantWithMinMaxVars,\
BoxPredictor_2/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,BoxPredictor_2/ClassPredictor/act_quant/FakeQuantWithMinMaxVars,\
BoxPredictor_3/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,BoxPredictor_3/ClassPredictor/act_quant/FakeQuantWithMinMaxVars,\
BoxPredictor_4/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,BoxPredictor_4/ClassPredictor/act_quant/FakeQuantWithMinMaxVars,\
BoxPredictor_5/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,BoxPredictor_5/ClassPredictor/act_quant/FakeQuantWithMinMaxVars" \
--inference_type=QUANTIZED_UINT8  --mean_values=128 --std_dev_values=128 --logtostderr
```

Please remember to rename the output model name to `ssd_mobilenet_v1/ssd_mobilenet_v1_quant/ssd_mobilenet_v2/ssd_mobilenet_v2_quant/ssdlite_mobilenet_v2.tflite` for example use.

Current WebML API doesn't support "Squeeze" operation. "Squeeze", "Reshape" and "Concatenation" operations are removed from graph because they are reduntant for inference. Use 6 box predictors and 6 class predictors from 6 feature maps for inference. [See tensorflow ssd_mobilenet_v1_feature_extractor for details.](https://github.com/tensorflow/models/blob/master/research/object_detection/models/ssd_mobilenet_v1_feature_extractor.py)

### For Tiny Yolo Models

#### Tiny Yolo COCO/VOC

This tflite model is converted from a keras model, which is converted from Darknet by [YAD2K](https://github.com/Wenzhao-Xiang/YAD2K).

For keras model generating, you can go [here](https://pjreddie.com/darknet/yolov2/) to download tiny yolo coco/voc weights and cfg file, and then follow the step of [YAD2K](https://github.com/Wenzhao-Xiang/YAD2K) to convert them to a keras model.

After get a keras model, you can then use the following commands to convert your own tflite model.

```sh
tflite_convert \
--keras_model_file=${model_dir}/${keras_model_name}.h5 \
--output_file=${out_dir}/${model_name}.tflite
```

Please remember to rename the output model name to `tiny_yolov2_coco/tiny_yolov2_voc.tflite` for example use.

## Note

The label files "coco_classes_part.txt" and "pascal.classes.txt" are both from [YAD2K](https://github.com/allanzelener/YAD2K), which are under `MIT LICENSE`.