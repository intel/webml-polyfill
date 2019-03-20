## How to Download
Download [ssd_mobilenet_v1](https://drive.google.com/file/d/1JlAXwCQztZ-ySmIQ8rZtJ0pncaPhCDm5/view?usp=sharing), [ssd_mobilenet_v2](https://drive.google.com/file/d/1JTotD3hmFL9ObHc-q-PlhBxYe3IqlQXH/view?usp=sharing),  [ssdlite_mobilenet_v2](https://drive.google.com/file/d/1YWDKpyUnMG6L4ddmt4wGGvOjx17e-9Fg/view?usp=sharing), [tiny_yolov2_coco](https://drive.google.com/file/d/1hosLKiVNiaDGV8QUGRwvrRC37oXKnLAr/view?usp=sharing), [tiny_yolov2_voc](https://drive.google.com/file/d/1fXksVZeVYsRyf_UnLDJf8-nkbCQmhW9J/view?usp=sharing), and put them here.

The model files are:

```txt
ssd_mobilenet_v1.tflite
ssd_mobilenet_v2.tflite
ssdlite_mobilenet_v2.tflite
tiny_yolov2_coco.tflite
tiny_yolov2_voc.tflite
```

## How to Generate

###  For SSD Mobilenet Models:

Check out [TensorFlow Lite Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for details.

These models are converted from [Tensorflow SSD MobileNet V1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz), [Tensorflow SSD MobileNet V2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) and [Tensorflow SSDLite MobileNet V2](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz). You can use the following commands to convert your own model.

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

Please remember to rename the output model name to `ssd_mobilenet_v1/ssd_mobilenet_v2/ssdlite_mobilenet_v2.tflite` for example use.

Current WebML API doesn't support "Squeeze" operation. "Squeeze", "Reshape" and "Concatenation" operations are removed from graph because they are reduntant for inference. Use 6 box predictors and 6 class predictors from 6 feature maps for inference. [See tensorflow ssd_mobilenet_v1_feature_extractor for details.](https://github.com/tensorflow/models/blob/master/research/object_detection/models/ssd_mobilenet_v1_feature_extractor.py)

###  For Tiny-Yolo Models:

#### Tiny-Yolo-COCO
Check out [kaka-lin/object-detection](https://github.com/kaka-lin/object-detection) for details.

This model is converted from [Tiny Yolo V2](https://drive.google.com/file/d/14-5ZojD1HSgMKnv6_E3WUcBPxaVm52X2/view?usp=sharing). You can use the following commands to convert your own model.

```sh
tflite_convert \
--keras_model_file=${download_model_dir}/tiny-yolo.h5 \
--output_file=${out_dir}/tiny_yolov2_coco.tflite
```

#### Tiny-Yolo-VOC
This tflite model is converted from a keras model, which is converted from Darknet by [YAD2K](https://github.com/allanzelener/YAD2K). 

For keras model generating, you can go [here](https://pjreddie.com/darknet/yolov2/) to download tiny-yolo-voc weight and cfg file, and then follow the step of [YAD2K](https://github.com/allanzelener/YAD2K) to convert them to a keras model.

After get a keras model, you can then use the following commands to convert your own tflite model.

```sh
tflite_convert \
--keras_model_file=${model_dir}/tiny-yolov2-voc.h5 \
--output_file=${out_dir}/tiny_yolov2_voc.tflite
```