const example = new ImageClassificationExample({model: imageClassificationModels});

const gallery = ['./img/0.jpg',
  './img/1.jpg',
  './img/2.jpg',
  './img/3.jpg',
  './img/4.jpg',
  './img/5.jpg',
  './img/6.jpg',
  './img/7.jpg',
  './img/8.jpg',
  './img/9.jpg',
  './img/10.jpg',
  './img/11.jpg',
]

$(document).ready(() => {
  example.UI();
  let gallerystring = ''
  for (let g of gallery) {
    let gstring = `<div class='gallery-item'><img class='gallery-image' src='${g}' alt=''></div>`
    gallerystring += gstring
  }
  $('#gallery').html(gallerystring)

  let cvpath = location.origin + location.pathname.replace('image_classification/', 'image_classification_opencv/')
  let nnpath = location.origin + location.pathname.replace('image_classification_opencv/', 'image_classification/')

  let cannedimage = location.search.replace('&s=camera', '&s=image')
  let video = location.search.replace('&s=image', '&s=camera')

  $('#cvimage').attr('href', nnpath + cannedimage)
  $('#cvcamera').attr('href', nnpath + video)

  $('#cvcannedimage').attr('href', cvpath + cannedimage)
  $('#cvvideo').attr('href', cvpath + video)

  $('#tabcvimage').click(function(){
    location.href = nnpath + cannedimage
  })

  $('#tabcvcamera').click(function(){
    location.href = nnpath + video
  })

  $('#tabcvcannedimage').click(function(){
    location.href = cvpath + cannedimage
  })

  $('#tabcvvideo').click(function(){
    location.href = cvpath + video
  })

  if(parseSearchParams('s') === 'camera') {
    $('#tabcvcannedimage').removeClass('active')
    $('#tabcvvideo').addClass('active')
    $('#imagetab').removeClass('active')
    $('#cameratab').addClass('active')
    $('#gallery').hide()
  }

  if(parseSearchParams('s') === 'image') {
    $('#tabcvcannedimage').addClass('active')
    $('#tabcvvideo').removeClass('active')
    $('#imagetab').addClass('active')
    $('#cameratab').removeClass('active')
    $('#gallery').fadeIn()
  }
});

$(window).load(() => {
  // Execute inference

  if(parseSearchParams('s') === 'image') {
    var time = 0
    $("#gallery .gallery-item").each(function() {
      var $this = $(this)
      setTimeout(function() {
        $("#gallery .gallery-item").removeClass('hl')
        $this.addClass('hl')
        let src = $this.children('img').attr('src')
        $('#feedElement').attr('src', src)
        example.main()
      }, time);
      time += 5000
    });
  } else {
    example.main()
  }
});
