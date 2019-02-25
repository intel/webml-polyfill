# Web Machine Learning (ML) API polyfill and examples


MacOS | Linux | Windows
-------- | -------- | --------
[![Build Status](https://api.travis-ci.com/intel/webml-polyfill.svg?branch=master)](https://travis-ci.com/intel/webml-polyfill) | [![CircleCI](https://circleci.com/gh/intel/webml-polyfill/tree/master.svg?style=svg)](https://circleci.com/gh/intel/webml-polyfill/tree/master) | [![Build status](https://ci.appveyor.com/api/projects/status/6xjudmjja1mcyo1m/branch/master?svg=true)](https://ci.appveyor.com/project/ibelem/webml-polyfill-egsl9/branch/master)


## Development / Testing

### Install

```sh
$ npm install
```

### Start

```sh
$ npm start
```

Start an HTTPS server:
```sh
$ HTTPS=true npm start
```

### Build

```sh
$ npm run build
```

Production build:

```sh
$ NODE_ENV=production npm run build
```

WASM backend build:

```sh
$ npm run build-wasm
```

### Test

```sh
$ npm start
```

Open browser and navigate to http://localhost:8080/test

### Watch

```sh
$ npm run watch
```

# License
This project is following [Apache License Version 2.0](./LICENSE_APACHE2).

And all documents in [test/wpt/resources](./test/wpt/resources) are licensed by the [W3C 3-clause BSD License](./test/wpt/resources/LICENSE).
