const example = new SpeechRecognitionExample({ model: speechRecognitionModels});

$(document).ready(() => {
  example.UI();
});

$(window).load(() => {
  // Execute inference
  example.main();
});
