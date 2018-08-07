let guiState;
function setupGui(){
  guiState = {
    model: 1.01,
    outputStride: 16,
    scaleFactor: 0.8,
    scoreThreshold: 0.5,
    multiPoseDetection: {
      nmsRadius: 20.0,
      maxDetections: 15,
    },
  };
  const gui = new dat.GUI({width: 300});
  gui.domElement.id = 'gui';
  gui.add(guiState, 'model', [0.5, 0.75, 1.0, 1.01]).onChange((model)=>{
    guiState.model = model;
    util._version = guiState.model;
    drawSingleandMulti();
  });
  gui.add(guiState, 'outputStride', [8, 16, 32]).onChange((outputStride)=>{
    guiState.outputStride = outputStride;
    util._outputStride = guiState.outputStride;
    drawSingleandMulti();
  });
  gui.add(guiState, 'scaleFactor', ).onChange((scaleFactor)=>{
    guiState.scaleFactor = scaleFactor;
    util._scaleFactor = guiState.scaleFactor;
    drawSingleandMulti();
  })
  gui.add(guiState, 'scoreThreshold', 0.0, 1.0).onChange((scoreThreshold)=>{
    guiState.scoreThreshold = scoreThreshold;
    util._minScore = guiState.scoreThreshold;
    drawResult();
  });
  const multiPoseDetection = gui.addFolder('Multi Pose Estimation');
  multiPoseDetection.open();
  multiPoseDetection.add(guiState.multiPoseDetection, 'nmsRadius', 0.0, 40.0).onChange((nmsRadius)=>{
    guiState.multiPoseDetection.nmsRadius = nmsRadius;
    util._nmsRadius = guiState.multiPoseDetection.nmsRadius;
    drawResult();
  });
  multiPoseDetection.add(guiState.multiPoseDetection, 'maxDetections')
    .min(1)
    .max(20)
    .step(1)
    .onChange((maxDetections)=>{
      guiState.multiPoseDetection.maxDetections = maxDetections;
      util._maxDetection = guiState.multiPoseDetection.maxDetections;
      drawResult();
    });
}

setupGui();