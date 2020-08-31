const example = new ImageClassificationExample({model: imageClassificationModels});

const gallery = ['./img/0.png',
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
  './img/12.jpg'
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

  $('#squeezenet_onnx').click(function(){
    location.href = cvpath + '?b=' + parseSearchParams('b') + '&m=squeezenet_onnx&s='+ parseSearchParams('s') +'&d=0&f=OpenCV.js'
  })

  $('#mobilenet_v2_onnx').click(function(){
    location.href = cvpath + '?b=' + parseSearchParams('b') + '&m=mobilenet_v2_onnx&s=' + parseSearchParams('s') +'&d=0&f=OpenCV.js'
  })

  $('#resnet_v1_onnx').click(function(){
    location.href = cvpath + '?b=' + parseSearchParams('b') + '&m=resnet_v1_onnx&s=' + parseSearchParams('s') +'&d=0&f=OpenCV.js'
  })

  $('#resnet_v2_onnx').click(function(){
    location.href = cvpath + '?b=' + parseSearchParams('b') + '&m=resnet_v2_onnx&s=' + parseSearchParams('s') +'&d=0&f=OpenCV.js'
  })

  if(parseSearchParams('s') === 'camera') {
    $('#tabcvcannedimage').removeClass('active')
    $('#tabcvvideo').addClass('active')
    $('#imagetab').removeClass('active')
    $('#cameratab').addClass('active')
    $('#gallery').hide()
    $('.inputf+label').show()
  }

  if(parseSearchParams('s') === 'image') {
    $('#tabcvcannedimage').addClass('active')
    $('#tabcvvideo').removeClass('active')
    $('#imagetab').addClass('active')
    $('#cameratab').removeClass('active')
    $('#gallery').fadeIn()
    $('.inputf+label').hide()
  }

  $("#gallery .gallery-item:first-child").hide()

});
