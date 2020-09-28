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

1. [facenet.bin](https://drive.google.com/file/d/15WCBhkck6NtVanLRUh3K-y9AUcgiGvha/view?usp=sharing) and [facenet.xml](https://drive.google.com/file/d/1dwDCzsMx_DmNJkoy4TG6E8g1kLF7xdUs/view?usp=sharing).
2. [face-reidentification-retail-0095.bin](https://drive.google.com/file/d/1wGlPqVPZDC9JL4ePNOUPp2BDeFgKQWTL/view?usp=sharing) and [face-reidentification-retail-0095.xml](https://drive.google.com/file/d/1XGFvIPtG929gdViGAWrF6P-JopuHK7Yy/view?usp=sharing).

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
