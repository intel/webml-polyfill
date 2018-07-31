let guiState;
function setupGui(){
  guiState = {
    algorithm: 'multi-pose',
    input: {
      model: 0.75,
      outputStride: 16,
      imageScaleFactor: 0.5,
      scoreThreshold: 0.5,
    },
    multiPoseDetection: {
      maxDetections: 5,
      nmsRadius: 30.0,
    },
  };

  const gui = new dat.GUI({width: 300});
  gui.domElement.id = 'gui';
  const algorithm = gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']).onChange((algorithm)=>{
    guiState.algorithm = algorithm;
    main();
  });
  const input = gui.addFolder('Input');
  input.open();
  input.add(guiState.input, 'model', [0.50, 0.75, 1.00, 1.01]).onChange((model)=>{
    guiState.input.model = model;
    main();
  });
  input.add(guiState.input, 'outputStride', [8, 16, 32]).onChange((outputStride)=>{
    guiState.input.outputStride = outputStride;
    main();
  });
  input.add(guiState.input, 'imageScaleFactor', 0.1, 1.0).onChange((scaleFactor)=>{
    guiState.input.imageScaleFactor = scaleFactor;
    main();
  });
  input.add(guiState.input, 'scoreThreshold', 0.0, 1.0).onChange((scoreThreshold)=>{
    guiState.input.scoreThreshold = scoreThreshold;
    main();
  });
  const multiPoseDetection = gui.addFolder('Multi Pose Estimation');
  multiPoseDetection.open();
  multiPoseDetection.add(guiState.multiPoseDetection, 'nmsRadius', 0.0, 40.0).onChange((nmsRadius)=>{
    guiState.multiPoseDetection.nmsRadius = nmsRadius;
    main();
  });
  multiPoseDetection.add(guiState.multiPoseDetection, 'maxDetections')
    .min(1)
    .max(20)
    .step(1)
    .onChange((maxDetections)=>{
      guiState.multiPoseDetection.maxDetections = maxDetections;
      main();
    });
}

setupGui();
