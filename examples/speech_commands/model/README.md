# Prerequisites

Download 
kws_cnn
and put them here.

The model files are:

```txt
kws_cnn.tflite
labels.txt
```

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