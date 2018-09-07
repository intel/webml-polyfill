let guiState;
guiState = {
  algorithm: 'multi-pose',
  model: 1.01,
  outputStride: 16,
  scaleFactor: 0.5,
  scoreThreshold: 0.5,
  multiPoseDetection: {
    maxDetections: 15,
    nmsRadius: 20.0,
  },
  showPose: true,
  showBoundingBox: false,
};
const gui = new dat.GUI({width: 300});
gui.domElement.id = 'gui';
const model = gui.add(guiState, 'model', [0.50, 0.75, 1.00, 1.01]);
const outputStride = gui.add(guiState, 'outputStride', [8, 16, 32]);
const scaleFactor = gui.add(guiState, 'scaleFactor', [0.25, 0.5, 0.75, 1.00]).listen();
const scoreThreshold = gui.add(guiState, 'scoreThreshold', 0.0, 1.0).listen();
const multiPoseDetection = gui.addFolder('Multi Pose Estimation');
multiPoseDetection.open();
const nmsRadius = multiPoseDetection.add(guiState.multiPoseDetection, 'nmsRadius', 0.0, 40.0);
const maxDetections = multiPoseDetection.add(guiState.multiPoseDetection, 'maxDetections')
  .min(1)
  .max(20)
  .step(1);
const showPose = gui.add(guiState, 'showPose');
const showBoundingBox = gui.add(guiState, 'showBoundingBox');
gui.close();
let customContainer = document.getElementById('my-gui-container');
customContainer.appendChild(gui.domElement);
