const app = new ObjectDetectionExample({model: objectDetectionModels});

$(document).ready(() => {
  app.UI();
});

$(window).load(() => {
  // Execute inference
  app.main();
});