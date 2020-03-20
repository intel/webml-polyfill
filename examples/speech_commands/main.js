const example = new speechCommandsExample({ model: speechCommandModels});

$(document).ready(() => {
  example.readyUI();
});

$(window).load(() => {
  // Execute inference
  example.mainAsync();
});
