## Get models
Download [ssd_mobilenetv1_face](https://drive.google.com/file/d/14hxWR1AFEQRfnExSm42R4ntYGX4QKMZz/view?usp=sharing), [ssd_mobilenetv2_face](https://drive.google.com/file/d/1mDDCJVZyaIZXz6bpJE_zF1C3nZCH45qD/view?usp=sharing),  [ssdlite_mobilenetv2_face](https://drive.google.com/file/d/1QAFPChUVU4MgQwQSO7n1ac0BGpQAmmBK/view?usp=sharing), [tiny_yolov2_face](https://drive.google.com/file/d/1PWW2LxKXPSlW-X4epEFITqJOYrgo3lep/view?usp=sharing). These models are used for face detection, please put them to facial_landmark_detection/model. And download [facenet.bin](https://drive.google.com/open?id=17HINkLRewWNOBGCK_bamPp2RweQHtv90) and [facenet.xml](https://drive.google.com/open?id=1qHom5LEc0K9Asoa4__68HT0g6L3g9cft). This model is used for face recognition, please download it here.

The model files are:

```txt
ssd_mobilenetv1_face.tflite
ssd_mobilenetv2_face.tflite
ssdlite_mobilenetv2_face.tflite
tiny_yolov2_face.tflite
facenet.{bin,xml}
```

## Convert `.pb` to `.{bin,xml}`

### Download the model source

Download [facenet](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) model.

### Download the convert tool

Download [open_model_zoo repo](https://github.com/opencv/open_model_zoo) and install [OpenVINO™ toolkit](https://docs.openvinotoolkit.org/latest/index.html) for the model converter.

### Convert this models

Follow this [downloader and converter](https://github.com/opencv/open_model_zoo/tree/master/tools/downloader) to convert the facenet model.

Get [open_model_zoo repo](https://github.com/opencv/open_model_zoo) and into `~/open_model_zoo/tools/downloader/`:

```sh
./converter.py \
--name 'facenet*' \
--download_dir ${facenet model dir} \
--output_dir ${output dir}
```
