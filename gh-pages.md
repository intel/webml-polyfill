# gh-pages

Due to some reasons like CDN, model locations, some source code need to be modified before commiting to gh-pages branch.

## Steps

### Build webml-polyfill.js

Build `webml-polyfill.js` to latest and upload to `dist` folder

### Amazon S3
Upgrade Web Machine Learning models to latest in Amason S3, accounts: @ibelem, @BruceDai, @huningxin

```
Cache-Control: public, max-age=2592000
```

Set Bucket Policy for `ARN: arn:aws:s3:::webnnmodel`

### Copy all files to gh-pages branch

### SW.js

Update `const CACHE_NAME = 'v19';` to newer like `v20`

### examples/index.html

Add follwing code for registering service worker in page footer before `</body>`

```
  <script>
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', function() {
        navigator.serviceWorker.register('/webml-polyfill/sw.js').then(function(registration) {
          // Registration was successful
          console.log('ServiceWorker registration successful with scope: ', registration.scope);
        }, function(err) {
          // registration failed :(
          console.log('ServiceWorker registration failed: ', err);
        });
      });
    }
  </script>
 ```

### examples/util/modelZoo.js 

 Replace `: '../`

 with 

 `: 'https://webnnmodel.s3-us-west-2.amazonaws.com/`


### examples/skeleton_detection/src/helperFunc.js in Line 33

Replace

```
  // const urlBase = 'https://storage.googleapis.com/tfjs-models/weights/posenet/';
  let urlBase = '../skeleton_detection/model/';
```

with

```
  const urlBase = 'https://webnnmodel.s3-us-west-2.amazonaws.com/skeleton_detection/model/';
  // const urlBase = '../skeleton_detection/model/';
```

### workload/resources/utils_sd.js in Line 90

Update 

```
    this.model = new PoseNet(modelArch, {'version': this.modelVersion, 'adjustPath': true,},
                             useAtrousConv, this.outputStride, this.scaleInputSize,
                             smType, cacheMap, backend, getPreferString());
```

to

```
     this.model = new PoseNet(modelArch, this.modelVersion,
                             useAtrousConv, this.outputStride, this.scaleInputSize,
                             smType, cacheMap, backend, getPreferString());
```
