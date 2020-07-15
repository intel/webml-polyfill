class BaseCameraExample extends BaseExample {
  constructor(models) {
    super(models);
    this._bFrontCamera = false;
  }

  /**
   * This method is to set '_bFrontCamera'.
   * @param {boolean|undefined} flag
   */
  useFrontFacingCamera = (flag) => {
    if (typeof flag == "undefined") {
      this._bFrontCamera = !this._bFrontCamera;
    } else {
      this._bFrontCamera = flag;
    }
  };

  /** @override */
  _getDefaultInputType = () => {
    return 'image';
  };

  /** @override */
  _getDefaultInputMediaType = () => {
    return 'camera';
  };

  /** @override */
  _getMediaStream = async () => {
    let stream;
    // The video source is facing toward the user
    let constraints = {audio: false, video: {facingMode: 'user'}};

    if (!this._bFrontCamera) {
      if (useBackCameraWorkaround) {
        // workaround for {facingMode: 'environment'} not working
        let deviceInfos = await navigator.mediaDevices.enumerateDevices();
        for (let i = 0; i !== deviceInfos.length; ++i) {
          const deviceInfo = deviceInfos[i];
          if (deviceInfo.kind === 'videoinput' && deviceInfo.label.indexOf('facing back') !== -1) {
            constraints = {audio: false, video: {deviceId: {exact: deviceInfo.deviceId}}}
            try {
              stream = await navigator.mediaDevices.getUserMedia(constraints);
              return stream;
            } catch (e) {
              console.error(`Failed to start video source by the video input: '${deviceInfo.label}'.`);
              // try next available video input
              continue;
            }
          }
        }
      } else {
        // The video source is facing away from the user, thereby viewing their environment
        constraints = {audio: false, video: {facingMode: 'environment'}};
      }
    }

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    return stream;
  };

  /** @override */
  _commonUIExtra = () => {
    if (/Mobi|Android|iPhone|iPad|iPod/.test(navigator.userAgent)) {
      // for mobile devices: smartphone, pad
      if (this._currentInputType === 'camera') {
        $('#cameraswitcher').show();
      } else {
        $('#cameraswitcher').hide();
      }

      $('#cameraswitch').prop('checked', this._bFrontCamera);

      $('#img').click(() => {
        $('#cameraswitcher').hide();
        this._setInputElement(this._feedElement);
      });

      $('#cam').click(() => {
        $('#cameraswitcher').fadeIn();
        this._setInputElement(this._feedMediaElement);
      });

      $('#cameraswitch').click(() => {
        $('.alert').hide();
        this.useFrontFacingCamera();
        $('#cameraswitch').prop('checked', this._bFrontCamera);
        this.main();
      });

      $('#fullscreen i svg').click(() => {
        $('#cameraswitcher').toggleClass('fullscreen');
      })
    } else {
      $('#cameraswitcher').hide();
    }

    $('#cam').click(() => {
      $('#fps').show();
    });

    $('#fullscreen i svg').click(() => {
      const toggleFullScreen = () => {
        let doc = window.document;
        let docEl = doc.documentElement;
        let requestFullScreen = docEl.requestFullscreen || docEl.mozRequestFullScreen || docEl.webkitRequestFullScreen || docEl.msRequestFullscreen;
        let cancelFullScreen = doc.exitFullscreen || doc.mozCancelFullScreen || doc.webkitExitFullscreen || doc.msExitFullscreen;
        if (!doc.fullscreenElement && !doc.mozFullScreenElement && !doc.webkitFullscreenElement && !doc.msFullscreenElement) {
          requestFullScreen.call(docEl);
        } else {
          cancelFullScreen.call(doc);
        }
      };

      $('#fullscreen i').toggle();
      toggleFullScreen();
      $('#overlay').toggleClass('video-overlay');
      $('#fps').toggleClass('fullscreen');
      $('#fullscreen i').toggleClass('fullscreen');
      $('#ictitle').toggleClass('fullscreen');
      $('#inference').toggleClass('fullscreen');
    });

    if (this._currentInputType == 'image') {
      this._currentInputElement.addEventListener('load', () => {
        this.main();
      }, false);
    }
  };
};
