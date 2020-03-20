const app = new objectDetectionExample({model: objectDetectionModels});

$(document).ready(() => {
  app.readyUI();
});

$(window).load(() => {
  // Execute inference
  app.mainAsync();
});