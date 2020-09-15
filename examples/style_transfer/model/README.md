## Training and Conversion

### 1. Prepare Environment

Clone [repo](https://github.com/acerwebai/VangoghCrazyWorld), install TensorFlow 1.14 and Pillow 3.4.2, scipy 0.18.1, numpy 1.11.2, ffmpeg 3.1.3 or later version.
```
$ git clone https://github.com/acerwebai/VangoghCrazyWorld.git
$ pip install -r requirements.txt
```

### 2. Convert Pre-Trained Model

Freeze model file, following the instruction that tensorflow bundled here:
```
python -m tensorflow.python.tools.freeze_graph \
--input_graph=tf-model/starrynight-300-255-NHWC_nbc8_bs1_7e00_1e03_0.001/graph.pbtxt \
--input_checkpoint=tf-model/starrynight-300-255-NHWC_nbc8_bs1_7e00_1e03_0.001/saver \
--output_graph=tf-models/starrynight.pb \
--output_node_names="output"
```
Convert TensorFlow `pb` model to `tflite` model:
```
tflite_convert \
--graph_def_file=tf-models/starrynight.pb \
--output_file=tf-models/starrynight.tflite \
--input_arrays=img_placeholder \
--output_arrays=add_37
```
Then you'll get a tflite model named `starrynight.tflite`.

### 3. Get Dataset & VGG19

Before training, you need get dataset from [COCO](http://images.cocodataset.org/zips/test2014.zip) and VGG19 from [matconvnet](http://www.vlfeat.org/matconvnet/), or execute `setup.sh` to get dataset and VGG19.
```
$ ./setup.sh
```

### 4. Training and Evaluating
You need create a folder "ckpts" in the root of this project to save chackpoint files. And you can change `--data-format` to get the data format you want.
```
$ python style.py 
  --data-format NHWC \
  --num-base-channels 4 \
  --style examples/style/starrynight-300-255.jpg \
  --checkpoint-dir ckpts \
  --test examples/content/farm.jpg \
  --test-dir examples/result \
  --content-weight 7e0 \
  --style-weight 1e3
  --checkpoint-iterations 1000 \
  --learning-rate 1e-3
  --batch-size 1
``` 
You can evaluate the trained models via:
```
$ python evaluate.py 
  --data-format NHWC \
  --num-base-channels 4 \
  --checkpoint tf-models/starrynight-300-255-NHWC_nbc4_bs1_7e00_1e03_0.01 \
  --in-path examples/content/farm.jpg \
  --out-path examples/results/
```

### 5. Convert `.meta` to `.onnx`

Install tf2onnx from pypi:
```
$ pip install -U tf2onnx
```
Convert TensorFlow models to ONNX:
```
$ python -m tf2onnx.convert \
  --checkpoint SOURCE_CHECKPOINT_METAFILE_PATH \
  --output TARGET_ONNX_MODEL.onnx \
  --inputs img_placeholder:0 \
  --outputs add_37:0 \
  --inputs-as-nchw img_placeholder:0 \
```
Then you'll get a onnx model named `TARGET_ONNX_MODEL.onnx`.
