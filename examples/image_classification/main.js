const example = new imageClassificationExample({model: imageClassificationModels});

$(document).ready(() => {
  example.readyUI();
});

$(window).load(() => {
  // Execute inference
  example.mainAsync();
});
