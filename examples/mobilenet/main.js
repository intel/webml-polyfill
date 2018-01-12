async function loadModelDataFile(url) {
  let response = await fetch(url);
  let arrayBuffer = await response.arrayBuffer();
  let bytes = new Uint8Array(arrayBuffer);
  return bytes;
}

var tfModel;
var model;
function main() {
  loadModelDataFile('./model/mobilenet_v1_1.0_224.tflite').then(bytes => {
    let buf = new flatbuffers.ByteBuffer(bytes);
    tfModel = tflite.Model.getRootAsModel(buf);
    //printTfLiteModel(tfModel);
    model = new MobileNet(tfModel);
    model.createCompiledModel().then(result => {
      console.log(`compilation result: ${result}`);
    }).catch(e => {
      console.error(e);
    })
  }).catch(e => {
    console.log(e);
  })
}
