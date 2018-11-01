# RegressionChecker
This is an automation tool kit to check regression easily for developers when submitting Web ML API PRs with high quality(avoiding new problems such as crash, freeze, etc.).

## Prerequisites
* Chromium build is required to be installed on the target device before the test
* For checking PRs relevant to Android platform, host pc needs install chrome or chromium browser firstly.

## Install
```sh
   $ npm install
```
   If installing `chromedriver` fails, you can install `chromedriver` with this command:

      $ npm install chromedriver --chromedriver_cdnurl=http://cdn.npm.taobao.org/dist/chromedriver

## Set Configurations
   There are two fields in the config.json, for example:
```
   {
     "platform": "Mac",
     "chromiumPath": "/User/test/Downloads/Chromium.app/Contents/MacOS/Chromium"
   }
```
   or
```
   {
     "platform": "Windows",
     "chromiumPath": "C:\\test\\win_x64_SUCCEED\\Chrome-bin\\chrome.exe"
   }
```
   You need modify these two fields for the different platforms:
   + **_platform_**: `{string}`, target platform, support **Android**, **Mac**, **Linux** and **Windows**
   + **_chromiumPath_**: `{string}`,**Mac**/**Linux**/**Windows**: the target chromium path   **Android**: the chrome or chromium path in above Prerequisites to show the final checking results

## Run Tests

```sh
$ npm start
```

## Support Platforms

|  Linux  |   Mac   |  Android  |  Windows  |
|  :---:  |  :---:  |   :---:   |   :---:   |
|  PASS   |   PASS  |    PASS   |    PASS   |

## Result html

![result-html](./baseline/result-html.png)
