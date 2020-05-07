const example = new ImageClassificationExample({model: imageClassificationModels});

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  // Execute inference
  example.main();
});
