const example = new SpeechCommandsExample({ model: speechCommandModels});

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  // Execute inference
  example.main();
});
