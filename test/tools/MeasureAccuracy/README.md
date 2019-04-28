Description
======
Currently this tool kit is only for internal use, it loads DeepLab v3+ (trained on PASCAL VOC 2012) in TFLite format, constructs and inferences batches of images by WebNN API, shows the average accuracy of inferenced images comparing GT images.

Prerequisite
-----------
Before launch this tool kit, you need to download the model. Please check out [README.md](model/README.md) in model folder for details.

Steps
-----------
1. Follow [README](https://github.com/intel/webml-polyfill/blob/master/README.md) to launch Http Server
2. Open browser and navigate to http://localhost:8080/test/tool/MeasureAccuracy
3. Click "Pick GT Image" to select GT images for comparing
4. Click "Pick Image" to select target images for inferencing, those selected images should pair with firstly GT images
5. Check "average accuracy" score on UI Page
