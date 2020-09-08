const fs = require('fs');
const path = require('path');
const http = require('http');
const url = require('url');
const cheerio = require('cheerio');
const crypto = require('crypto');
const settings = require("../settings.js");
const util = require("../lib/Util.js");

const buildInfo = settings.NIGHTLY_BUILD_INFO[util.getTargetPlatform()];
let htmlElement;

const getMD5Online = (md5URL) => {
  let md5Value = "";
  http.get(md5URL, (res) => {
    const {statusCode} = res;
    let error;
    if (statusCode == 404) {
      throw new Error("Please make sure md5 file exist!");
    }
    if (statusCode !== 200) {
      error = new Error("Request Failed.\n" + `Status Code: ${statusCode}`);
    }
    if (error) {
      console.log(error.message);
      res.resume();
      return;
    }
    res.on("data", (chunk) => {
      md5Value += chunk;
    });
    res.on("end", () => {
      md5Value = md5Value.split(" ")[0];
      util.saveLatestBuildInfo('md5', md5Value);
    });
  }).on("error", (e) => {
    console.log(`getMD5Online func got error: ${e.message}`);
  });
}

const checkMD5 = (path) =>{
  if (!fs.existsSync(path)) {
    throw new Error(`No such '${path}' file for MD5 checking.`);
  }

  let valueMD5 = crypto.createHash("md5").update(fs.readFileSync(path)).digest("hex");
  return valueMD5 === util.getMD5();
};

const getHtmlELE = async (URL) => {
  // sorted build commits by 'Last modified' column of descending order
  URL += "/?C=M;O=A";
  return new Promise ((resolve, reject) => {
    let html;
    let options = {
      host: url.parse(URL).host,
      path: url.parse(URL).path,
      port: 80
    }
    htmlElement = [];
    http.get(options, (res) => {
      res.on("data", (data) => {
        html += data;
      });
      res.on("end", () => {
        let allHtmlELE = cheerio.load(html);
        resolve(allHtmlELE);
      });
    }).on("error", (err) => {
      console.log(`getHtmlELE func got error: ${err.message}`);
    });
  });
};

const getTargetCommit = async () => {
  let downloadCommit;
  if (settings.TARGET_BUILD_COMMIT !== "latest") {
    await getHtmlELE(settings.NIGHTLY_BUILD_URL).then((ele) => {
      ele('a').each((i, e) => {
        htmlElement.push(ele(e).attr("href").split("/")[0]);
      });
    });
    if (htmlElement.indexOf(settings.TARGET_BUILD_COMMIT) !== -1) {
      downloadCommit = settings.TARGET_BUILD_COMMIT;
    } else {
      console.log("Target build commit is invalid, please specify correct 'TARGET_BUILD_COMMIT' in 'settings.js' file.");
    }
  } else {
    await getHtmlELE(settings.NIGHTLY_BUILD_URL).then((ele) => {
      // get latest commit id
      downloadCommit = ele("a")[ele("a").length-1]["attribs"]["href"].slice(0, -1);
    });
    console.log(`1.1-Got latest commit as '${downloadCommit}'.`);
    util.saveLatestBuildInfo("latestcommit", downloadCommit);
  }
  return downloadCommit;
};

const getChromiumName = async () => {
  let targetCommit = settings.TARGET_BUILD_COMMIT === "latest" ? util.getLatestCommit() : settings.TARGET_BUILD_COMMIT;
  let downloadPath = settings.NIGHTLY_BUILD_URL + targetCommit + "/" + buildInfo["path"] + "/";
  let chromiumPackageName;
  await getHtmlELE(downloadPath).then((ele) => {
    ele('a').each((i,e) => {
      htmlElement.push(ele(e).attr("href").split("/")[0]);
    });
    String.prototype.endWith = function (endStr) {
      let d = this.length - endStr.length;
      return (d >= 0 && this.lastIndexOf(endStr) == d);
    }
    htmlElement.forEach((data) => {
      if (data.endWith(buildInfo["suffix"])) {
        util.saveLatestBuildInfo("chromiumbuild", data);
        chromiumPackageName = data;
      }
    });
  });
  await getMD5Online(downloadPath + chromiumPackageName + ".md5");
  return chromiumPackageName;
};

(async () => {
  console.log(`>>> 1-Start download nightly build at ${(new Date()).toLocaleTimeString()}.`);
  let downloadCommit = await getTargetCommit();
  let downloadChromiumPath = settings.NIGHTLY_BUILD_URL + downloadCommit + "/" + buildInfo["path"] + "/";
  let downloadPackageName = await getChromiumName();
  let storeFileLocation = path.join(util.getChromiumPath(), buildInfo["path"], downloadPackageName);

  if (fs.existsSync(storeFileLocation)) {
    if (await checkMD5(storeFileLocation)) {
      console.log(`Target build of ${downloadCommit} has already been downloaded, no need download again!`);
    } else {
      fs.unlinkSync(storeFileLocation);
      console.log("Failed to MD5 check local package, please download again!");
      process.exit(1);
    }
  } else {
    console.log(`1.2-Now downloading build from ${downloadChromiumPath}${downloadPackageName}, please wait ...`);
    await util.downloadChromium(downloadChromiumPath + downloadPackageName);
  }
})().catch((err) => {
  throw err;
});