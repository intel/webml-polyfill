const example = new emotionAnalysisExample({emotionAnalysis: emotionAnalysisModels,
                                            faceDetection: faceDetectionModels});
// use front facing camera
example.setFrontCameraFlag(true);

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  // Execute inference
  example.main();
});
