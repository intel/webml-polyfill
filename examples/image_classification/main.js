const example = new ImageClassificationExample({model: imageClassificationModels});

$(document).ready(() => {
  example.UI();

  let f = parseSearchParams('f')
  if(f.toLowerCase() === 'opencv.js') {
    $('#opencvspecial').html('Special Offer:')
    console.log('vvv')
  } else {
    console.log('dddd')
  }
});

$(window).load(() => {
  // Execute inference
  example.main();
});
