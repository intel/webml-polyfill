## PerLayerAnalyzer
This is an automation tool kit to automatically test the every layer of realmodel testcase and generate the display results on the page.

## Prerequisites
* Chromium build is required to be installed on the target device before the test.
* For test Android platform, host pc needs install chrome or chromium browser firstly.

## Here Are Two Ways To Run This TOOL :

#### The First Way ( Depends on our local server environment )

* If you want to use the environment we've already deployed, you can run the following command 
```sh
   $  git clone https://github.com/intel/webml-polyfill.git
```
```sh
   $  cd webml-polyfill/test/tools/perLayerAnayzer
```
```sh
   $  npm install
```

* Then change Settings in the **webml-polyfill/test/tools/perLayerAnayzer/config.json** fill.

* Finally, execute the command
```sh
   $  npm start
```

#### The Second Way ( Create your own server and environment )

* First you need generate the realmodel related testcase locally according to **TWO modelNames**, you can click on this [link](https://github.com/intel/webml-polyfill/blob/master/test/realmodel/README.md) ( Make sure the server always open )

* You need to Remove **'/webml-polyfill'** from **webml-polyfill/test/tools/perLayerAnayzer/src/main.js** in **line 916,918,922,924** . Like: 
```javascript
   remoteURL = `http://${urlServer}/test/squ_realmodel.html`
```

* Finally, change the config.json file like first way and execute the command
```sh
   $  cd webml-polyfill/test/tools/perLayerAnayzer
```
```sh
   $  npm install
```
```sh
   $  npm start
```

## Install
```sh
   $  npm install
```
   You need modify chromedriver version to '2.45.0' in package.json when you run chromium 70 build. chromedriver 2.46.0 supports chromium >=71.

   If installing `chromedriver` fails, you can install `chromedriver` with this command:

      $ npm install chromedriver --chromedriver_cdnurl=http://cdn.npm.taobao.org/dist/chromedriver

## Set Configurations
   There are eight fields in the config.json, for example:
```
   {
     "urlServer": "IP:8080",
     "modelName": ["squeezenet1.1"],
     "platform": "Mac",
     "chromiumPath": "/Users/test/Downloads/Chromium.app/Contents/MacOS/Chromium",
     "supportSwitch": false,
     "API": "polyfill",
     "preference": "fast"
     "iterations": "200"
   }
```
   or

```
   {
      "urlServer": "IP:8080",
      "modelName": ["squeezenet1.1"],
      "platform": "Linux",
      "chromiumPath": "/usr/bin/chromium-browser-unstable",
      "supportSwitch": false,
      "API": "polyfill",
      "preference": "fast",
      "iterations": "200"
   }
```
   or

```
   {  
     "urlServer": "IP:8080",
     "modelName": ["squeezenet1.1"],
     "platform": "Windows",
     "chromiumPath": "C:\\test\\win_x64_SUCCEED\\Chrome-bin\\chrome.exe",
     "API": "polyfill",
     "preference": "fast",
     "supportSwitch": false,
     "iterations": "200"
   }
```
   You need modify these eight fields for the different platforms:
   + **_urlServer_**: `{string}`, Server IP address, port number 8080.
   + **_modelName_**: `{array}`, We support **squeezenet1.1**, **mobilenetv2-1.0** two models, you need choose **["squeezenet1.1"]** , **["mobilenetv2-1.0"]** , **["squeezenet1.1", "mobilenetv2-1.0"]** , **["mobilenetv2-1.0", "squeezenet1.1"]** .
   + **_platform_**: `{string}`, target platform, support **Android**, **Mac**, **Linux** and **Windows**.
   + **_chromiumPath_**: `{string}`, **Mac**/**Linux**/**Windows**: the target chromium path **Android**: the chrome or chromium path in above Prerequisites to show the final checking results.
   + **_supportSwitch_**: `{boolean}`, support **true** and **false**.
   + **_API_**: `{string}`, choose to **polyfill** and **webnn**.
   + **_preference_**: `{string}`, choose to  **fast** , **sustained** and **low**.
   + **_iterations _**: `{string}`, set the number of times you want to run the realmodel case.

|    |  Platform  |  Fast  |  Sustained  |  Low  |
|  :-----:  |  :----:  |   :----:   |   :----:   |   :----:   |
|  swith:true API:webnn  |  Windows   |    MKLDNN   |    DirectML   |   DirectML   |
|  swith:true API:webnn  |  macOS  |  MKLDNN  |  MPS  |      |
|  swith:true API:webnn  |  Linux  |   IE-MKLDNN	   |   IE-clDNN	   |   IE-MYRIAD  |
|  swith:false API:webnn  |  Windows  |  MKLDNN  |  clDNN  |      |
|  swith:false API:webnn  |  macOS  |   BNNS   |   MPS   |      |
|  swith:false API:webnn  |  Linux   |    MKLDNN   |    clDNN   |      |
|  swith:false API:polyfill  |  Windows  |   WASM   |   WebGL   |      |
|  swith:false API:polyfill  |  macOS  |   WASM   |   WebGL   |      |
|  swith:false API:polyfill  |  Linux  |   WASM   |   WebGL   |      |


## Support Platforms

|  Linux  |   Mac   |  Android  |  Windows  |
|  :---:  |  :---:  |   :---:   |   :---:   |
|  PASS   |   PASS  |    PASS   |    PASS   |


## How TO Use Template.html File ( If you use one of the above ways, you don`t  need follow these steps )

* In the near future, We w add new modelName testcase. You can import names of your new generated realmodel testcase file in the **line 54-57** of **webml-polyfill/test/template.html** file. Like:
```javascript
   <script src="./realmodel/testcase/squeezenet1.1/squeezenet1.1-conv2d-1.js"></script>
```
* Then you should chang the name of **template.html** by modelname

|  ModelName  |   OldFileName   |  NewFileName  |
|  :-----  |  :-----:  |   :-------   |
|  squeezenet1.1   |   template.html  |    real_squeezenet1.1.html   |
|  mobilenetv2-1.0  |   template.html   |  real_mobilenetv2-1.0.html  |
|  squeezenet1.1, mobilenetv2-1.0   |  template.html  |   real_squeezenet1.1_mobilenetv2-1.0.html   |
|  mobilenetv2-1.0, squeezenet1.1   |   template.html  |    real_mobilenetv2-1.0_squeezenet1.1.html  |