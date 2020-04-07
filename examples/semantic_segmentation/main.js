const example = new semanticSegmentationExample({model: semanticSegmentationModels});
// use front facing camera
example.useFrontFacingCamera(true);

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  example.loadedUI();
  // Execute inference
  example.main();
});
