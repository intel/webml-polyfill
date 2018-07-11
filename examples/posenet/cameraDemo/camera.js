async function main(){
  const video = document.getElementById('video');
  let streaming  = false;
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  let stats = new Stats();
  let util = new Utils();
  function isAndroid() {
    return /Android/i.test(navigator.userAgent);
  }

  function isiOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
  }

  function isMobile() {
    return isAndroid() || isiOS();
  }


  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  stats.dom.style.cssText = 'position:fixed;top:60px;right:10px;cursor:pointer;opacity:0.9;z-index:10000';
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video':{
      facingMode: 'user',
      width: mobile? undefined: videoWidth,
      height: mobile? undefined : videoHeight,
    },
  });

  let e = document.getElementById("backend");
  let backend = e.options[e.selectedIndex].text;
  switch(backend){
      case "WebGL":
          if (nnPolyfill.supportWebGL2){
                util.init('WebGL2').then(()=>{
                  predict();
                });
          }
          else{
              throw new Error("Do not support WebGL");
          }
          break;
      case "WASM":
          if (nnPolyfill.supportWasm){
              util.init('WASM').then(()=>{
                predict();
              });
          }
          else{
              throw new Error("Do not support WASM");
          }
          break;
      case "WebML":
          if(nnNative){
              util.init('WebML').then(()=>{
                predict();
              });
          }
          else{
              throw new Error("Do not support WebML");
          }
          break;
      default:
          break;
  }

  function loadVideo(){
    video.srcObject = stream;
    return new Promise((resolve, reject) =>{
      video.onloadedmetadata = () =>{
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
        streaming = true;
        resolve(video);
      };
    });
  }

  async function predict(){
    stats.begin();
    const videoElement = await loadVideo();
    util.predict();
    stats.end();
    if(streaming){
      setTimeout(predict, 0);
    }
  }
}

main();
