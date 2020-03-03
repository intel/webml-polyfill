const fs = require('fs');
const path = require('path');

async function checkURL(URL) {
  let reg = (/(((^https?:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+ )((?:\/[\+~%\/.\w-_]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)$/g);
  if (!reg.test(URL)) {
    return false;
  } else {
    return true;
  }
}

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

class getJsonData {
  constructor() {
    this.json;
    this.open();
  }

  open() {
    let filePath = path.join(__dirname, '..', 'config.json');
    this.json = JSON.parse(fs.readFileSync(filePath));
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

  getModelName() {
    if (this.json.modelName == null) {
      throw new Error('modelName is null');
    } else {
      return this.json.modelName;
    }
  }

  close() {
    fs.closeSync(0);
    fs.closeSync(1);
  }
}
JSON_DATA = new getJsonData();
getURL = JSON_DATA.getURL();
getlocalURL = JSON_DATA.getlocalURL();
getModelName = JSON_DATA.getModelName();

module.exports = {
  checkURL,
  mkdirsSync
};
