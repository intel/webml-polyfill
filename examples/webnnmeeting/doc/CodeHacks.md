There are some code hacks for running the app better.

## Intel Open WebRTC Toolkit (OWT)

Update `assets/js/owt/conference/channel.js` code

```
  if (options.audio === undefined) {
    options.audio = !!stream.settings.audio;
  }
```
to following for fixing a screen sharing mode issue caused by audio option:

```
  if (options.audio === undefined) {
    if(stream.settings.audio.length === 0) {
      options.audio = false
    } else {
      options.audio = !!stream.settings.audio;
    }
  }
```

Added code below in `createMediaStream()` of `assets/js/owt/base/mediastream-factory.js` for supporting Echo Cancellation and Noise Suppression standard Web API.

```
  if (constraints.audio.echoCancellation) {
    mediaConstraints.audio.echoCancellation = constraints.audio.echoCancellation;
  }
  if (constraints.audio.noiseSuppression) {
    mediaConstraints.audio.noiseSuppression = constraints.audio.noiseSuppression;
  }
```

## Material Design Icons

Modified Material Design Icons style sheet url in npm package for faster loading due to network issue. No necessary to follow this step if you don't encounter the issue.

Update `.node_modules/nuxt-buefy/modules.js` from `materialDesignIconsHRef: '//cdn.materialdesignicons.com/2.4.85/css/materialdesignicons.min.css'`
to `materialDesignIconsHRef: '../../css/materialdesignicons/2.4.85/materialdesignicons.min.css'`
