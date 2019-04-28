require('../lib/jsonOperation.js');
const fs = require('fs');
const path = require('path');
const targz = require('targz');

(async function() {
  async function Unzip(PATH) {
    let unzipPath = path.join(__dirname, '..', 'model');
    if (!fs.existsSync(PATH)) throw new Error(`${PATH} is not exist`);
    if (!fs.existsSync(unzipPath)) await mods.mkdirsSync(unzipPath);
    targz.decompress({
      src: PATH,
      dest: unzipPath
    }, (err) => {
      if (err) {
        throw new Error(`unzip ${PATH} fail`);
      }else {
      let savePath = path.join(__dirname, '..', 'model', `${JSON_DATA.getModelName()}`);
      var readDir = fs.readdirSync(savePath);
      for (var i=0;i< readDir.length;i++){
        value = readDir[i];
        reg = /.onnx/;
        match = reg.test(value);
        if (match) {
        var a = `${JSON_DATA.getModelName()}.onnx`;
        var regex = new RegExp(a);
        matchFlat = regex.test(value);
        if (!matchFlat){
          oldPath = path.join(__dirname, '..', 'model', `${JSON_DATA.getModelName()}`, value,)
          newPath = path.join(__dirname, '..', 'model', `${JSON_DATA.getModelName()}`, `${JSON_DATA.getModelName()}.onnx`,)
          fs.renameSync(oldPath,newPath);
          };
        }
        }
        };
    });
  }
  await Unzip(path.join(__dirname, '..', 'model', JSON_DATA.getURL().split('/').pop()));
})().then(() => {
  console.log('Unzip Module File' + ` ${JSON_DATA.getModelName()}.tar.gz`);

}).catch((err) => {
  throw err;
});