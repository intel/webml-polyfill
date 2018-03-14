function main() {
  let utils = new Utils();
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const checkboxElement = document.getElementById('GPU');

  checkboxElement.addEventListener('click', function(e) {
    if (checkboxElement.checked && nn.supportWebGL2) {
      utils.model = new MobileNet(utils.tfModel);
      utils.model.createCompiledModel( { useWebGL2: true }).then(result => {
        console.log(`compilation result: ${result}`);
        utils.predict(imageElement);
      }).catch(e => {
        console.error(e);
      })
    } else {
      utils.model = new MobileNet(utils.tfModel);
      utils.model.createCompiledModel( { useWebGL2: false }).then(result => {
        console.log(`compilation result: ${result}`);
        utils.predict(imageElement);
      }).catch(e => {
        console.error(e);
      })
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
  });
}
