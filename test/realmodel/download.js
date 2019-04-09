const fs = require('fs');
const path = require('path');
const wget = require('node-wget');
const targz = require('targz');
const puppeteer = require('puppeteer');

class getJsonData {
  constructor() {
    this.json;
    this.open();
  }

  open() {
    this.json = JSON.parse(fs.readFileSync('./config.json'));
  }

  getURL() {
    if (this.json.url == null) {
      throw new Error('url is null');
    } else {
      if (checkURL(this.json.url)) {
        return this.json.url;
      } else {
        throw new Error(`${this.json.url} it's not a standard url`);
      }
    }
  }

  getlocalURL() {
    if (this.json.localURL == null) {
      throw new Error('url is null');
    } else {
      return this.json.localURL;
    }
  }

  getModelPath() {
    if (this.json.modelPath == null) {
      throw new Error('modelPath is null');
    } else {
      return this.json.modelPath;
    }
  }

  getModelName() {
    if (this.json.modelName == null) {
      throw new Error('modelName is null');
    } else {
      return this.json.modelName;
    }
  }

  getJsonPath() {
    if (this.json.JSONPath == null) {
      throw new Error('JsonPath is null');
    } else {
      return this.json.JSONPath;
    }
  }

  getCasePath() {
    if (this.json.casePath == null) {
      throw new Error('casePath is null');
    } else {
      return this.json.casePath;
    }
  }

  getCaseSource() {
    if (this.json.caseSource == null) {
      throw new Error('caseSource is null');
    } else {
      return this.json.caseSource;
    }
  }

  close() {
    fs.closeSync(0);
    fs.closeSync(1);
  }
}
JSON_DATA = new getJsonData();

async function mkdirsSync(dirname) {
  if (fs.existsSync(dirname)) {
    return true;
  } else {
    if (mkdirsSync(path.dirname(dirname))) {
      fs.mkdirSync(dirname);
      return true;
    }
  }
}

async function checkURL(URL) {
  let reg = (/(((^https?:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+ )((?:\/[\+~%\/.\w-_]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)$/g);
  if (!reg.test(URL)) {
    return false;
  } else {
    return true;
  }
}

async function wgetDownload(URL) {
  let savePath = path.join(process.cwd(), 'tool', 'model');
  await mkdirsSync(savePath);
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

async function Unzip(PATH) {
  let unzipPath = path.join(process.cwd(), 'tool', 'model');
  if (!fs.existsSync(PATH)) throw new Error(`${PATH} is not exist`);
  if (!fs.existsSync(unzipPath)) await mkdirsSync(unzipPath);

  targz.decompress({
    src: PATH,
    dest: unzipPath
  }, (err) => {
    if (err) {
      throw new Error(`unzip ${PATH} fail`);
    } else {
      console.log(`Finished unzip ${PATH} to ${unzipPath} !`);
    }
  });
}

async function downloadFile(URL, savePath) {
  const browser = await puppeteer.launch({headless: false});
  savePath = path.join(process.cwd(), 'tool');
  if (fs.existsSync(path.join(savePath, 'casePrototypeData.json'))) {
    fs.unlinkSync(path.join(savePath, 'casePrototypeData.json'));
  }
  try {
    const page = await browser.newPage();
    await page.goto(URL);
    await page._client.send('Page.setDownloadBehavior', {
      behavior: 'allow',
      downloadPath: savePath,
    });
    await page.waitFor(30000);
  } finally{
    await browser.close();
  }
}

(async function() {
  await wgetDownload(JSON_DATA.getURL());
  await downloadFile(JSON_DATA.getlocalURL(), process.cwd());
  await Unzip(path.join(process.cwd(), 'tool', 'model', JSON_DATA.getURL().split('/').pop()));
})();
