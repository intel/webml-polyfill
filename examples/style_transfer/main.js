const example = new StyleTransferExample({model: styleTransferModels});

$(document).ready(() => {
  // $('.photo-scrollbar').first().hide()
  example.UI();
});

$(window).load(() => {
  // Execute inference
  let text = `<div class="vg">
  <div>the painting style of <span>Van Gogh<span></div><br>
  <strong>The Starry Night</strong>
  </div>`;
  $('#stname').html(text);
  example.main();
});
