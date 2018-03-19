function main() {
  let utils = new Utils();
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const checkboxElement = document.getElementById('WebGL2');
  const checkboxLable = document.getElementById('checkboxLable');

  checkboxElement.addEventListener('click', function(e) {
    if (checkboxElement.checked && nn.supportWebGL2) {
      utils.model = new MobileNet(utils.tfModel);
      utils.model.createCompiledModel( { useWebGL2: true }).then(result => {
        utils.predict(imageElement, true).then(result => {
          utils.predict(imageElement, false);
        });
      }).catch(e => {
        console.error(e);
      });
    } else {
      utils.model = new MobileNet(utils.tfModel);
      utils.model.createCompiledModel( { useWebGL2: false }).then(result => {
        utils.predict(imageElement);
      }).catch(e => {
        console.error(e);
      });
    }
  });

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
    if (!nn.supportWebGL2) {
      checkboxElement.setAttribute('hidden', true);
      checkboxLable.innerHTML = 'Do not support WebGL2!';
      checkboxLable.setAttribute('class', 'alert alert-warning');
    }
  });
}
