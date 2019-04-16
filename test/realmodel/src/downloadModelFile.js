require('../lib/jsonOperation.js');
const path = require('path');
const wget = require('node-wget');
const mods = require('../lib/jsonOperation.js');

(async function() {
  async function wgetDownload(URL) {
    let savePath = path.join(__dirname,'..', 'model');
    await mods.mkdirsSync(savePath);
    let name = path.join(savePath, JSON_DATA.getURL().split('/').pop());
    await wget({
      url: URL,
      dest: name,
      timeout: 30000
    }, (error) => {
      if (error) {
        throw new Error(`Download func get error: ${error}`);
      }
    });
  }
  await wgetDownload(JSON_DATA.getURL());
})().then(() => {
  console.log('Downloading Module File...');
}).catch((err) => {
  throw err;
});
