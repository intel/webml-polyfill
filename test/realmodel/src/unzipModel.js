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
      }
    });
  }
  await Unzip(path.join(__dirname, '..', 'model', JSON_DATA.getURL().split('/').pop()));
})().then(() => {
  console.log('Unzip Module File...');
}).catch((err) => {
  throw err;
});
