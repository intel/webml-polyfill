## How to Download
Download [ssd_mobilenetv1_face](https://drive.google.com/file/d/14hxWR1AFEQRfnExSm42R4ntYGX4QKMZz/view?usp=sharing), [ssd_mobilenetv2_face](https://drive.google.com/file/d/1mDDCJVZyaIZXz6bpJE_zF1C3nZCH45qD/view?usp=sharing),  [ssdlite_mobilenetv2_face](https://drive.google.com/file/d/1QAFPChUVU4MgQwQSO7n1ac0BGpQAmmBK/view?usp=sharing), [tiny_yolov2_face](https://drive.google.com/file/d/1PWW2LxKXPSlW-X4epEFITqJOYrgo3lep/view?usp=sharing). These models are used for face detection, please put them to facial_landmark_detection/model. And Download [emotion_classification](https://drive.google.com/file/d/14ZKigyyqlCpu6CPHDGJ8nKVGAxrF8JAO/view?usp=sharing). This model is used for emotion analysis, please put it here.

The model files are:

```txt
ssd_mobilenetv1_face.tflite
ssd_mobilenetv2_face.tflite
ssdlite_mobilenetv2_face.tflite
tiny_yolov2_face.tflite
emotion_classification_7.tflite
```

## How to Generate

### For Face Landmark Detection Models

Check out [ratnajitmukherjee/EmotionClassification_FER2013](https://github.com/ratnajitmukherjee/EmotionClassification_FER2013) for more details about this model.

This model is converted from a pre-trained [CNN model](https://drive.google.com/file/d/1MNsaiLrlSxjlZkrnvyZRdXQGESY3luoH/view?usp=sharing). You can use the following commands to convert your own model.

```sh
tflite_convert \
--keras_model_file=${download_model_dir}/emotion_classification_7classes.hdf5 \
--output_file=${out_dir}/emotion_classification_7.tflite
```
