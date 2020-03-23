const app = new objectDetectionExample({model: objectDetectionModels});

$(document).ready(() => {
  app.UI();
});

$(window).load(() => {
  // Execute inference
  app.main();
});