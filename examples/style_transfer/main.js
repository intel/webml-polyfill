const example = new StyleTransferExample({model: styleTransferModels});

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  // Execute inference
  example.main();
});
