# Prerequisites

Download [KWS_DNN](https://drive.google.com/drive/folders/1b5MANOiJCSt0N9YVHCtMny_UsaGfQAuE?usp=sharing) and [KWS_CNN](https://drive.google.com/open?id=195qDe5xaz6VmIegg5OPE6Ih7UinWVhnf) then put them here.

The model files are:

```
kws_cnn.tflite
kws_dnn.bin
kws_dnn.xml
labels.txt
labels2.txt
```

# KWS_DNN

This model is pre-trained by [ARM-software](https://github.com/ARM-software/ML-KWS-for-MCU/tree/master/Pretrained_models/DNN) and  then converted to OpenVINO IR format. Steps below show how to convert the model to OpenVINO IR format.

## Download and Conversion

### 1. Clone the ARM-software/ML-KWS-for-MCU project
```
git clone https://github.com/ARM-software/ML-KWS-for-MCU.git
```

### 2. Install Intel OpenVINO Toolkit
Download [Intel OpenVINO Toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download?elq_cid=5653349&erpm_id=8668281).

Install OpenVINO Toolkit follow the [guide](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#install-openvino) and enable OpenVINO env.

### 3. Convert `.pb` to `.bin` & `.xml`
```
cd ML-KWS-for-MCU/Pretrained_models/DNN
mo_tf.py --input_model DNN_S.pb \
--input  Reshape:0 \
--input_shape [1,250] \
--output add \
--disable_nhwc_to_nchw \
--output_dir kws_dnn
```

You will get DNN_S.xml and DNN_S.bin in the kws_dnn folder. Rename them:
```
mv DNN_S.xml kws_dnn.xml
mv DNN_S.bin kws_dnn.bin
```
# KWS_CNN

This model is trained by [tensorflow/examples](https://github.com/tensorflow/examples/tree/master/lite/examples/speech_commands/ml) and then converted to TfLite format. Steps below show how to train and convert the model to tflite format.

## Training and Conversion

### 1. Prepare Environment
Python 3.5+  
Keras 2.1.6 or higher  
pandas and pandas-ml  
TensorFlow 1.5 or higher

Clone [repo](https://github.com/tensorflow/examples/tree/master/lite/examples/speech_commands), install dependance.

```
$ git clone https://github.com/tensorflow/examples.git
$ pip install -r requirements.txt
```

### 2. Download data

Run the download script to load the dataset into the local filesystem.

```
python download.py
```

### 3. Training

The model can be trained by running train.py

```
python train.py -sample_rate 16000 -batch_size 64 -output_representation raw -data_dirs data/train
```

You will get a output folder with `.hdf5` models.

### 4. Convert `.hdf5` to `.pb`

The models can be convert to `.pb` model.

```
python export/convert_keras_to_quantized.py
```

You will get xxx.hdf5.pb.

### 5. Convert `.pb` to `.tflite`

You can change the output node.

```
tflite_convert --output_file kws_cnn.tflite \
--graph_def_file xxx.hdf5.pb \
--output_format TFLITE \
--inference_type FLOAT \
--inference_input_type FLOAT \
--input_arrays input_1 \
--output_arrays conv1d_14/add \

```
