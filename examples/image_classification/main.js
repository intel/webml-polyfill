const example = new ImageClassificationExample({model: imageClassificationModels});

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  if (example._currentFramework === 'WebNN') {
    // Execute inference
    example.main();
  }
});
