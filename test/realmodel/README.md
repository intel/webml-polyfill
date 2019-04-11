# Generate realmodel testcase tool 

## Prerequisites
* Follow [README](https://github.com/intel/webml-polyfill/blob/master/README.md), to launch HTTP Server for WebML.

### Install
```sh
$ npm install
```

### All
```sh
$ npm run all
```

### Download Model
```sh
$ npm run downloadModel
```

### Unzip Model
```sh
$ npm run unzipModel
```

### Get case origin data
```sh
$ npm run getCaseOriginData
```

### Generate case
```sh
$ npm run generateCase
```

## Set config.json file

* `modelName`: model name.
* `url`: download model file url.
* `localURL`: url for local index.html.
* `backend`: backend value.

## Output
* `./testcase`: auto generated folder for saving testcase files.
* `./testcase/res`: auto generated folder for resource files.
* `./model`: auto generated folder for saving model files.
