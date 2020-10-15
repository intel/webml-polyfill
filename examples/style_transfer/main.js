const example = new StyleTransferExample({model: styleTransferModels});

$(document).ready(() => {
  // $('.photo-scrollbar').first().hide()
  example.UI();
});

$(window).load(() => {
  // Execute inference
  $('#stname').html('The starry night');
  example.main();
});
