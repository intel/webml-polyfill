## How to Download
Download [ssd_mobilenetv1_face](https://drive.google.com/file/d/14hxWR1AFEQRfnExSm42R4ntYGX4QKMZz/view?usp=sharing), [ssd_mobilenetv2_face](https://drive.google.com/file/d/1mDDCJVZyaIZXz6bpJE_zF1C3nZCH45qD/view?usp=sharing),  [ssdlite_mobilenetv2_face](https://drive.google.com/file/d/1QAFPChUVU4MgQwQSO7n1ac0BGpQAmmBK/view?usp=sharing), [tiny_yolov2_face](https://drive.google.com/file/d/1PWW2LxKXPSlW-X4epEFITqJOYrgo3lep/view?usp=sharing), [face_landmark](https://drive.google.com/file/d/1VNzrTRLgZNJoSgNxsfO__9GAxqQ0Je1Z/view?usp=sharing), and put them here.

The model files are:

```txt
ssd_mobilenetv1_face.tflite
ssd_mobilenetv2_face.tflite
ssdlite_mobilenetv2_face.tflite
tiny_yolov2_face.tflite
face_landmark.tflite
```

## How to Generate

### For SSD Mobilenet Face Detection Models

These ssd mobilenet face detection models are trained by Tensorflow Object Detection API with WIDER_FACE dataset. Please go [here](https://github.com/Wenzhao-Xiang/face-detection-ssd-mobilenet) for more training details.

After getting the `frozen_inference_graph.pb`, you can use the following commands to convert them to tflite models.

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

Please remember to rename the output model name to `ssd_mobilenetv1_face/ssd_mobilenetv2_face/ssdlite_mobilenetv2_face.tflite` for example use.

Current WebML API doesn't support "Squeeze" operation. "Squeeze", "Reshape" and "Concatenation" operations are removed from graph because they are reduntant for inference. Use 6 box predictors and 6 class predictors from 6 feature maps for inference. [See tensorflow ssd_mobilenet_v1_feature_extractor for details.](https://github.com/tensorflow/models/blob/master/research/object_detection/models/ssd_mobilenet_v1_feature_extractor.py)

### For Tiny-Yolo Face Detection Models

#### Tiny-Yolo-Face

Check out [abars/YoloKerasFaceDetection](https://github.com/abars/YoloKerasFaceDetectionn) for more details about this model.

This model is converted from [Tiny Yolo V2 Face](https://drive.google.com/file/d/1S-Yo1VXLzA9dPuKDVOLL6qQFuUBNlHXm/view?usp=sharing). You can use the following commands to convert your own model.

```sh
tflite_convert \
--keras_model_file=${download_model_dir}/yolov2_tiny-face.h5 \
--output_file=${out_dir}/tiny_yolov2_face.tflite
```

### For Face Landmark Detection Models

Check out [yinguobing/cnn-facial-landmark](https://github.com/yinguobing/cnn-facial-landmark) for more details about this model.

This model is converted from a pre-trained [Simple CNN](https://drive.google.com/file/d/1Nvzu5A9CjP70sDhiRbMzuIwFLnrq2Qpw/view?usp=sharing) model. You can use the following commands to convert your own model.

```sh
tflite_convert \
--output_file=${out_dir}/face_landmark.tflite \
--graph_def_file=${download_model_dir}/SimpleCNN.pb \
--input_arrays=input_to_float \
--output_arrays=logits/BiasAdd
```
