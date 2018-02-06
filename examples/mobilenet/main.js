function main() {
  let utils = new Utils();
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
    }
  }, false);

  imageElement.onload = function() {
    utils.predict(imageElement);
  }
  utils.init().then(() => {
    utils.predict(imageElement);
    button.setAttribute('class', 'btn btn-primary');
    input.removeAttribute('disabled');
  });
}
