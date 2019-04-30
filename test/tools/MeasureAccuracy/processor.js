let processor = {
  timerCallback: function() {
    if (this.video.paused || this.video.ended) {
      this.i = 1;
      return;
    }
    this.computeFrame();
    let self = this;
    setTimeout(function () {
        self.timerCallback();
      }, 0);
  },

  doLoad: function() {
    this.video = document.getElementById("video1");
    this.c1 = document.getElementById("c1");
    this.ctx1 = this.c1.getContext("2d");
    this.c2 = document.getElementById("c2");
    this.ctx2 = this.c2.getContext("2d");
    this.i = 1;
    let self = this;
    this.video.crossOrigin = "Anonymous";
    this.video.addEventListener("play", function() {
        self.width = self.video.videoWidth;
        self.height = self.video.videoHeight;
        self.timerCallback();
      }, false);
  },

  computeFrame: function() {
    this.c1.width = this.width;
    this.c1.height = this.height;
    this.ctx1.drawImage(this.video, 0, 0, this.width, this.height);
    console.log('time:', this.i);
    this.i += 1;
    // let frame = this.ctx1.getImageData(0, 0, this.width, this.height);
    //     let l = frame.data.length / 4;

    // for (let i = 0; i < l; i++) {
    //   let r = frame.data[i * 4 + 0];
    //   let g = frame.data[i * 4 + 1];
    //   let b = frame.data[i * 4 + 2];
    //   if (g > 100 && r > 100 && b < 43)
    //     frame.data[i * 4 + 3] = 0;
    // }
    // this.ctx2.putImageData(frame, 0, 0);
    // return;
  }
};

document.addEventListener("DOMContentLoaded", () => {
processor.doLoad();
});