const example = new faceRecognitionExample({faceRecognition: faceRecognitionModels,
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