async function main(){
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const scaleCanvas = document.getElementById('canvas_2');
  const scaleCtx = scaleCanvas.getContext('2d');
  const util = new Utils();
  const videoWidth = 500;
  const videoHeight = 500;
  const inputSize = [1, videoWidth, videoHeight, 3];
  const isMultiple = guiState.algorithm;
  let streaming  = false;	
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }
  let stats = new Stats();
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
        util.init('WebGL2', inputSize).then(()=>{
          predict();
          document.getElementById('loading').style.display = 'none';
          document.getElementById('camera').style.display = 'block';
        });
      }
      else{
        throw new Error("Do not support WebGL");
      }
      break;
    case "WASM":
      if (nnPolyfill.supportWasm){
        util.init('WASM', inputSize).then(()=>{
          predict();
          document.getElementById('loading').style.display = 'none';
          document.getElementById('camera').style.display = 'block';
        });
      }
      else{
        throw new Error("Do not support WASM");
      }
      break;
    case "WebML":
      if(nnNative){
        util.init('WebML', inputSize).then(()=>{
          predict();
          document.getElementById('loading').style.display = 'none';
          document.getElementById('camera').style.display = 'block';
        });
      }
      else{
        throw new Error("Do not support WebML");
      }
      break;
    default:
      break;
  }
  
  function isAndroid(){
    return /Android/i.test(navigator.userAgent);
  }
  
  function isiOS(){
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
  }
  
  function isMobile(){
    return isAndroid() || isiOS();
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
    await util.predict(scaleCanvas, ctx, inputSize);
    if(isMultiple == "multi-pose"){
    	util.drawOutput(canvas, 'multi', inputSize);
    }
    else{
      util.drawOutput(canvas, 'single', inputSize);
    }
    stats.end();
    if(streaming){
      setTimeout(predict, 100);
    }
  }
}

main();
