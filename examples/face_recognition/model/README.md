This example have tow parts: face detection and face recognition.

## Get models

### Face Detection Model
Download following models and put them to `facial_landmark_detection/model`:

1. [ssd_mobilenetv1_face](https://drive.google.com/file/d/14hxWR1AFEQRfnExSm42R4ntYGX4QKMZz/view?usp=sharing)
2. [ssd_mobilenetv2_face](https://drive.google.com/file/d/1mDDCJVZyaIZXz6bpJE_zF1C3nZCH45qD/view?usp=sharing)
3. [ssdlite_mobilenetv2_face](https://drive.google.com/file/d/1QAFPChUVU4MgQwQSO7n1ac0BGpQAmmBK/view?usp=sharing)
4. [tiny_yolov2_face](https://drive.google.com/file/d/1PWW2LxKXPSlW-X4epEFITqJOYrgo3lep/view?usp=sharing)

The model files are:

```txt
ssd_mobilenetv1_face.tflite
ssd_mobilenetv2_face.tflite
ssdlite_mobilenetv2_face.tflite
tiny_yolov2_face.tflite
```

### Face Recognition Model
Download and convert following models here:

1. [facenet.bin](https://drive.google.com/open?id=17HINkLRewWNOBGCK_bamPp2RweQHtv90) and [facenet.xml](https://drive.google.com/open?id=1qHom5LEc0K9Asoa4__68HT0g6L3g9cft).
2. [face-reidentification-retail-0095.bin](https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin) and [face-reidentification-retail-0095.xml](https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml).

The model files are:

```txt
facenet.{bin,xml}
face-reidentification-retail-0095.{bin,xml}
```

## Convert `.pb` to `.{bin,xml}`

### Download the model source

Download [facenet](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) model and [OpenVino repo](https://github.com/opencv/open_model_zoo) here.

### Convert this models

Follow this [downloader and converter](https://github.com/opencv/open_model_zoo/tree/master/tools/downloader) to convert the facenet model.
