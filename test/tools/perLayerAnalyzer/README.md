## PerLayerAnalyzer
This is an automation tool kit to automatically test the realmodel and generate the display results on the page.

## Prerequisites
* Chromium build is required to be installed on the target device before the test
* For checking PRs relevant to Android platform, host pc needs install chrome or chromium browser firstly.

## Here Are Two Ways To Run This TOOL()

### The One Way(Depends on our server environment):

1. If you want to use the environment we've already deployed,you can go directly to git clone
**$ git clone https://github.com/intel/webml-polyfill.git** and **$ cd  webml-polyfill/test/tools/perLayerAnayzer/** directory.
2. Then you can type **$ npm install** in Terminal and change some Settings in the **webml-polyfill/test/tools/perLayerAnayzer/config.json** fill of the current directory,

See **## Install** and **## Set Configurations** for more tips.
3. Finally,execute the command **$ npm start** in the current directory.See **## Run Tests** for more tips.

### The Second Way(Create your own server and environment):

1. First of all,you generate the realmodel related testcase locally according to **TWO modelNames**,you can click this [link](https://github.com/intel/webml-polyfill/blob/master/test/realmodel/README.md) (Make sure the server always open.)

2. You need to modify the code of **webml-polyfill/test/tools/perLayerAnayzer/src/main.js** in **line 916,918,922,924** like : remoteURL = `http://${urlServer}/test/squ_realmodel.html`;(Remove **webml-polyfill** from inside.)

3. Finally you can perform  **the One Way** steps two and three.

## Install
```sh
   $ npm install
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
      "preference": "fast"
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
     "preference": "fast"
     "supportSwitch": false,
     "iterations": "200"
   }
```
   You need modify these eight fields for the different platforms:
   + **_urlServer_**: `{string}`,  Server IP address,port number 8080(if you use our environment ,Server IP you can ask the developer concerned).
   + **_modelName_**: `{array}`, There are three options **["squeezenet1.1"]**, **["mobilenetv2-1.0"]**, **["squeezenet1.1", "mobilenetv2-1.0"]** to dispaly the model data you want.
   + **_platform_**: `{string}`, target platform, support **Android**, **Mac**, **Linux** and **Windows**.
   + **_chromiumPath_**: `{string}`, **Mac**/**Linux**/**Windows**: the target chromium path   **Android**: the chrome or chromium path in above Prerequisites to show the final checking results.
   + **_supportSwitch_**: `{boolean}`,support **true** and **false**.
   + **_API_**: `{string}`, choose to  **polyfill** , **webnn**.
   + **_preference_**: `{boolean}`, choose to  **fast** , **sustained**, **low**.
   + **_iterations _**: `{number}`, set the number of times you want to run the realmodel case.

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

  
## Run Tests

```sh
$ npm start
```

## Support Platforms

|  Linux  |   Mac   |  Android  |  Windows  |
|  :---:  |  :---:  |   :---:   |   :---:   |
|  PASS   |   PASS  |    PASS   |    PASS   |


##  How TO Use Template.html File (If you use one of the above ways,you don`t follow these steps.)

* In the near future ,We will add new modelName testcase. then , you can use **webml-polyfill/test/template.html** file to what you want to run.
* Please Follow These Steps:

1. You can  import your  generated realmodel testcase file name in the **line 54** of **webml-polyfill/test/template.html** file in the current directory.

For example : **<script src="./realmodel/testcase/squeezenet1.1/squeezenet1.1-conv2d-1.js"></script>**  in the code of **webml-polyfill/test/tools/perLayerAnayzer/template.html**.

you can **CP** to the **modelName.HTML** files generated by the realmodel, for example : **line 33-71** of **webml-polyfill/test/squeezenet1.1.html**.

2. if you want to test **squeezenet1.1**,you should change the **template.html** file name to **real_squeezenet1.1.html**.

if you want to test **mobilenetv2-1.0**,you should change the **template.html** file name to **real_mobilenetv2-1.0.html**.

if you want to test **mobilenetv2-1.0** and **squeezenet1.1**,you should change the **template.html** file name to **mob_realmodel.html**.

if you want to test **squeezenet1.1** and **mobilenetv2-1.0** ,you should change the **template.html** file name to **squ_realmodel.html**.