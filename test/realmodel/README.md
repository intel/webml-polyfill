# Generate realmodel testcase tool 

## Prerequisites
* Follow [README](https://github.com/intel/webml-polyfill/blob/master/README.md), to launch HTTP Server for WebML.

### Install
```sh
$ npm install
```

### Download
```sh
$ npm run download
```

### Generate case
```sh
$ npm run gentc
```

## Set config.json file

* `modelName`: model name.
* `url`: download model file url.
* `localURL`: url for local index.html.

## Output
* `./testcase`: auto generated folder for saving testcase files.
* `./testcase/res`: auto generated folder for resource files.
* `./tool/model`: auto generated folder for saving model files.
