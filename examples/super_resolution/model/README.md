# Prerequisites

Download 
[srgan_96_4](https://drive.google.com/file/d/1WLbsFWhlSWF2FPFSwu5e7WkpXvzv2TYc/view?usp=sharing),
[srgan_128_4](https://drive.google.com/file/d/1aZwhsLeXIEW4P62ncoDZ9dpsdaqG8Oym/view?usp=sharing),
and put them here.

The model files are:

```txt
srgan_96_4.tflite
srgan_128_4.tflite
```

These models are trained by [TensorLayer](https://github.com/tensorlayer/tensorlayer) and then converted to TfLite format. Steps below show how to train and convert the model to tflite format.

## Training and Conversion

### 1. Prepare Environment

Clone [repo](https://github.com/GreyZzzzzzXh/srgan), install TensorFlow and TensorLayer.

```
$ git clone https://github.com/GreyZzzzzzXh/srgan.git
$ pip3 install TensorFlow TensorLayer
```

### 2. Modify SubpixelConv2d

[Tensorlayer.layers.SubpixelConv2d()](https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/layers/convolution/super_resolution.py#L114) uses tf.depth_to_space(), which is not supported in tflite. 

Go to the definition of SubpixelConv2d() in your local file, and replace `tf.depth_to_space()` with `tf.transpose()` and `tf.batch_to_space_nd()`.

```py
# X = tf.depth_to_space(X, r)
X = tf.transpose(X, [3, 1, 2, 0])
X = tf.batch_to_space_nd(X, [r, r], [[0, 0], [0, 0]])
X = tf.transpose(X, [3, 1, 2, 0])
```

See https://github.com/GreyZzzzzzXh/srgan/blob/master/model.py#L49.

### 3. Train and Evaluate 

Train and evaluate model according to [srgan/README.md](https://github.com/GreyZzzzzzXh/srgan/blob/master/README.md). You will get some checkpoint files in srgan/checkpoint.

### 4. Freeze Graph

```
$ cd srgan/checkpoint
$ python3 -m tensorflow.python.tools.freeze_graph \
--input_graph=graph.pb \
--input_checkpoint=model.ckpt \
--input_binary=true \
--output_graph=frozen_graph.pb \
--output_node_names=SRGAN_g/out/Tanh
```
You will get frozen_graph.pb.

### 5. Convert `.pb` to `.tflite`

You can change the input size.

```
$ toco \
--input_file=frozen_graph.pb \
--output_file=srgan_96_4.tflite \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--input_shapes=1,96,96,3 \
--input_arrays=input_image \
--output_arrays=SRGAN_g/out/Tanh \
--inference_type=FLOAT \
--logtostderr
```

Then you'll get a tflite model named srgan_96_4.tflite.