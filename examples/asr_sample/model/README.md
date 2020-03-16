# Inter OpenVINO

Intel [OpenVINO](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_speech_sample_README.html) shows how to run the speech sample application, which demonstrates acoustic model inference based on Kaldi* neural networks and speech feature vectors.

### 1. Prepare Environment
OpenVINO

Install OpenVINO toolkit follow the [Install Guides](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html).

### 2. Download wsj_dnn5b_smbr model
Download the pre-trained model from https://download.01.org/openvinotoolkit/models_contrib/speech/kaldi/wsj_dnn5b_smbr/ or using the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader).

### 3. Convert model
You can use the following model optimizer command to convert a Kaldi nnet1 or nnet2 neural network to Intel IR format:
```
$ python3 mo.py --framework kaldi --input_model wsj_dnn5b.nnet --counts wsj_dnn5b.counts --remove_output_softmax
```
Assuming that the model optimizer (mo.py), Kaldi-trained neural network, wsj_dnn5b.nnet, and Kaldi class counts file, wsj_dnn5b.counts, are in the working directory this produces the Intel IR network consisting of wsj_dnn5b.xml and wsj_dnn5b.bin.

You will get wsj_dnn5b.xml and wsj_dnn5b.bin and move them here.