const example = new imageClassificationExample({model: imageClassificationModels});

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  // Execute inference
  example.main();
});
