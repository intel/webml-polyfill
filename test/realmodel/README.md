# Generate realmodel testcase tool 

1. Download and unzip [Squeezenet](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz) to `tool/generate_realmodel/model/`
2. Copy `tool/generate_realmodel/index.html` to `../../examples/util/onnx/` and browser it with chrome, it will auto download json file to local.
3. Enter `tool/generate_realmodel/` run command `node main.js ***.json` (second step download json file as arguments).
4. Test case will generate in `testcase/`, data file will save in `testcase/res` folder.
