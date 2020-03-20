class baseCameraExample extends baseExample {
  constructor(models) {
    super(models);
    this._bFrontCamera = false;
  }

  setFrontCameraFlag = (flag) => {
    if (typeof flag == "undefined") {
      this._bFrontCamera = !this._bFrontCamera;
    } else {
      this._bFrontCamera = flag;
    }
  };

  _getMediaConstraints = () => {
    const constraints = {audio: false, video: {facingMode: (this._bFrontCamera ? 'user' : 'environment')}};
    return constraints;
  };

  _readyCommonUIExtra = () => {
    if (/Mobi|Android|iPhone|iPad|iPod/.test(navigator.userAgent)) {
      // for mobile devices: smartphone, pad
      if (this._currentInputElement.tagName == 'VIDEO') {
        $('#cameraswitcher').show();
      } else {
        $('#cameraswitcher').hide();
      }

      $('#cameraswitch').prop('checked', this.bIsFrontCamera);

      $('#img').click(() => {
        $('#cameraswitcher').hide();
        const element = document.getElementById('feedElement');
        this._setInputElement(element);
      });

      $('#cam').click(() => {
        $('#cameraswitcher').fadeIn();
        const element = document.getElementById('feedMediaElement');
        this._setInputElement(element);
      });

      $('#cameraswitch').click(() => {
        $('.alert').hide();
        this.setFrontCameraFlag();
        $('#cameraswitch').prop('checked', this.bIsFrontCamera);
        this.mainAsync();
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

    if (this._currentInputElement.tagName == 'IMG') {
      this._currentInputElement.addEventListener('load', () => {
        this.mainAsync();
      }, false);
    }
  };
};
