async function loadModelDataFile(url) {
  let response = await fetch(url);
  let bytes = await response.arrayBuffer();
  return bytes;
}

loadModelDataFile('model_data.bin').then(bytes => {
  let simpleModel = new SimpleModel(bytes);
  simpleModel.createCompiledModel();
  simpleModel.compute(1, 1).then(result => {
    console.log(`result: ${result}`);
  }).catch(error => {
    console.log(error);
  })
});