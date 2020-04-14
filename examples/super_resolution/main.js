const example = new SuperResolutionExample({model: superResolutionModels});

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  // Execute inference
  example.main();
});
