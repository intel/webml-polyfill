require('../lib/jsonOperation.js');
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async function() {
  async function downloadFile(URL) {
    const browser = await puppeteer.launch({headless: false});
    let savePath = path.join(__dirname, '..', 'model', JSON_DATA.getModelName());
    if (fs.existsSync(path.join(savePath, `${JSON_DATA.getModelName()}.json`))) {
      fs.unlinkSync(path.join(savePath, `${JSON_DATA.getModelName()}.json`));
    }
    try {
      const page = await browser.newPage();
      await page.goto(URL);
      await page._client.send('Page.setDownloadBehavior', {
        behavior: 'allow',
        downloadPath: savePath,
      });
      await page.waitFor(120000);
    } finally{
      await browser.close();
    }
  }
  await downloadFile(JSON_DATA.getlocalURL());
})().then(() => {
  console.log('Downloading Case Data...');
}).catch((err) => {
  throw err;
});
