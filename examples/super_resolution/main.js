const example = new superResolutionExample({model: superResolutionModels});

$(document).ready(() => {
  example.readyUI();
});

$(window).load(() => {
  // Execute inference
  example.mainAsync();
});
