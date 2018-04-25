Download [ssd_mobilenet](https://drive.google.com/file/d/1bKD4eK8Zh9x_R7wc9CxpLHk2hrYG5orU/view?usp=sharing) and unzip here.

The model and label files are:

```txt
coco_labels_list.txt
ssd_mobilenet.tflite
```

Check out [TensorFlow Lite Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for details.

This modle is conver from [Tensorflow SSD MobileNet model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz). You can use the following commands to convert your own model.

```sh
python ${tensorflow_dir}/lib/python3.5/site-packages/tensorflow/python/tools/optimize_for_inference.py \
--input=${download_dir}/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb \
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

toco \
--input_file=${out_dir}/frozen_inference_graph_stripped.pb \
--output_file=${out_dir}/ssd_mobilenet.tflite \
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