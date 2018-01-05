async function loadModelDataFile(url) {
  let response = await fetch(url);
  let bytes = await response.arrayBuffer();
  return bytes;
}

loadModelDataFile('model_data.bin').then(bytes => {
  let simpleModel = new SimpleModel(bytes);
  simpleModel.createCompiledModel().then(result => {
    console.log(`compilation result: ${result}`)
    simpleModel.compute(1, 1).then(result => {
      console.log(`execution result: ${result}`);
    }).catch(error => {
      console.log(error);
    });
  });
});