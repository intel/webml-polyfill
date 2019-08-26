# PerLayerAnalyzer
This is an automation tool kit to automatically test the realmodel and generate the display results on the page

## Prerequisites
* Chromium build is required to be installed on the target device before the test
* For checking PRs relevant to Android platform, host pc needs install chrome or chromium browser firstly.

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
     "supportSwitch": false,S
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
   + **_urlServer_**: `{string}`,  Server IP address,port number 8080.
   + **_modelName_**: `{array}`, There are three options **["squeezenet1.1"]**, **["mobilenetv2-1.0"]**, **["squeezenet1.1", "mobilenetv2-1.0"]** to dispaly the model data you want.
   + **_platform_**: `{string}`, target platform, support **Android**, **Mac**, **Linux** and **Windows**.
   + **_chromiumPath_**: `{string}`, **Mac**/**Linux**/**Windows**: the target chromium path   **Android**: the chrome or chromium path in above Prerequisites to show the final checking results.
   + **_supportSwitch_**: `{boolean}`, **Mac**: `--use-mkldnn`, **Linux**: `--use-inference-engine`, **Windows**: `--use-dml`, support **true** and **false**.
   + **_API_**: `{string}`, choose to  **polyfill** , **webnn**.
   + **_preference_**: `{boolean}`, choose to  **fast** , **sustained**, **low**.
   + **_iterations _**: `{number}`, set the number of times you want to run the realmodel case.
## Run Tests

```sh
$ npm start
```

## Support Platforms

|  Linux  |   Mac   |  Android  |  Windows  |
|  :---:  |  :---:  |   :---:   |   :---:   |
|  PASS   |   PASS  |    PASS   |    PASS   |

## Special Description

If you want to deploy locally to run ,you can use the template html file.On  **line 54** of this file ,you can import your generated realmodel case file name here . What modelname do you need for this modelname generated caes file use  **<src**in template.html.