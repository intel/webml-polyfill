const example = new speechCommandsExample({ model: speechCommandModels});

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  // Execute inference
  example.main();
});
