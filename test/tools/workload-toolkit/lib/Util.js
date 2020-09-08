const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const url = require('url');
const extract = require('extract-zip');
const childProcess = require('child_process');
const csv = require('fast-csv');
const moment = require('moment');
const settings = require("../settings.js");

const util = {
  mkdirsSync: function (dirname) {
    if (fs.existsSync(dirname)) {
      return true;
    } else {
      if (this.mkdirsSync(path.dirname(dirname))) {
        fs.mkdirSync(dirname);
        return true;
      }
    }
  },

  deleteFolder: function (dirname) {
    let files = [];
    if(fs.existsSync(dirname)) {
      files = fs.readdirSync(dirname);
      files.forEach(function(files, index) {
        let curPath = dirname + path.sep + files;
        if(fs.statSync(curPath).isDirectory()) { // recurse
          this.deleteFolder(curPath);
        } else { // delete file
          fs.unlinkSync(curPath);
        }
      });
      fs.rmdirSync(dirname);
    }
  },

  getTargetPlatform: function () {
    let platform;
    if (os.type() === "Linux") {
      platform = "Linux";
    } else if (os.type() === "Windows_NT") {
      platform = "Windows";
    } else {
      throw new Error(`This toolkit currently only supports for testing on Linux and Windows platforms.`);
    }

    return platform;
  },

  getRealTargetBackend: function () {
    let realTargetBackend = settings.TARGET_BACKEND;
    if (this.getTargetPlatform() === 'Linux') {
      realTargetBackend = realTargetBackend.filter(function (item) {return item !== 'DirectML';});
    }

    return realTargetBackend;
  },

  getLatestCommit: function () {
    let infoJson = JSON.parse(fs.readFileSync("./latest-commit.json"));
    return infoJson.latestcommit;
  },

  getMD5: function () {
    let infoJson = JSON.parse(fs.readFileSync("./latest-commit.json"));
    return infoJson.md5;
  },

  getChromiumBuild: function () {
    let infoJson = JSON.parse(fs.readFileSync("./latest-commit.json"));
    return infoJson.chromiumbuild;
  },

  saveLatestBuildInfo: function (key, value) {
    let infoJson = JSON.parse(fs.readFileSync("./latest-commit.json"));
    infoJson[key] = value;
    fs.writeFileSync("./latest-commit.json", JSON.stringify(infoJson, null, 2));
  },

  getChromiumPath: function () {
    let chromiumPath;
    if (settings.TARGET_BUILD_COMMIT !== 'latest') {
      chromiumPath = path.join(process.cwd(), "output", "chromiumBuild", settings.TARGET_BUILD_COMMIT);
    } else {
      chromiumPath = path.join(process.cwd(), "output", "chromiumBuild", this.getLatestCommit());
    }
    this.mkdirsSync(chromiumPath);
    return chromiumPath;
  },

  downloadChromium: function (buildURL) {
    let platform = this.getTargetPlatform();
    let savepath = path.join(this.getChromiumPath(), settings.NIGHTLY_BUILD_INFO[platform]["path"]);
    this.mkdirsSync(savepath);
    let name = path.join(savepath, this.getChromiumBuild());
    let options = {
      host: url.parse(buildURL).host,
      path: url.parse(buildURL).pathname,
      port: 80
    }
    let files = fs.createWriteStream(name);
    http.get(options, (res) => {
      res.on("data", (data) => {
        files.write(data);
      });
      res.on("end", () => {
        files.end();
      });
    }).on("error", (err) => {
      console.log(`download func got error: ${err.message}`);
    })
  },

  uninstallChromium: function ()  {
    if (this.getTargetPlatform() === "Linux") {
      console.log("2.1-First uninstall existed chromium package if there was.");
      let command = "echo '" + settings.LINUX_PASSWORD + "' | sudo -S dpkg -r chromium-browser-unstable";
      let subprocess = childProcess.execSync(
        command, {timeout: 300000, encoding: "UTF-8", stdio: [process.stdin, process.stdout, "pipe"]});
    }
  },

  installChromium: function (installPath) {
    console.log(`installPath ---- ${installPath}`);
    if (!fs.existsSync(installPath)) {
      throw new Error(`No such '${installPath}' file for installing package`);
    }

    let target_platform = this.getTargetPlatform();
    if (target_platform === "Linux") {
      let command = "echo '" + settings.LINUX_PASSWORD + "' | sudo -S dpkg -i " + installPath;
      childProcess.execSync(
        command,
        {stdio: [process.stdin, process.stdout, "pipe"], timeout: 300000}
      );
    } else if (target_platform === "Windows") {
      let unzipPath = path.join(this.getChromiumPath(), settings.NIGHTLY_BUILD_INFO[target_platform]["path"]);
      extract(installPath, {dir: unzipPath});
    }
  },

  getReportPath: function () {
    let commit = settings.TARGET_BUILD_COMMIT;
    if (settings.TARGET_BUILD_COMMIT === 'latest') {
      commit = this.getLatestCommit();
    }

    let reportPath = path.join(process.cwd(), "output", "report", commit);
    util.mkdirsSync(reportPath);
    return reportPath;
  },

  openCSV: async function (flag) {
    let realTargetBackend = this.getRealTargetBackend();
    let csvStream = await csv.createWriteStream({headers: true})
      .transform(function (row) {
        let csvColumns = {
          "CATEGORY": row.category,
          "MODEL": row.model
        };
        for (let backend of realTargetBackend) {
          csvColumns[backend] = row[backend.toLocaleLowerCase()];
        }
        return csvColumns;
      });

    let targetPlatform = this.getTargetPlatform().toLocaleLowerCase();
    let csvName;

    if (flag) {
      let devReportPath = path.join(process.cwd(), "output", "report", "dev");
      this.mkdirsSync(devReportPath);
      csvName = devReportPath + path.sep + "result-" + targetPlatform + ".csv";
    } else {
      if (settings.REGRESSION_FLAG) {
        let baselineReportPath = path.join(process.cwd(), "output", "report", "baseline");
        this.mkdirsSync(baselineReportPath);
        csvName = baselineReportPath + path.sep + "baseline-" + targetPlatform + ".csv";
      } else {
        csvName = this.getReportPath() + path.sep + "result-" + targetPlatform +  "-" + moment().format("YYYYMMDDHHmmsss") + ".csv";
      }
    }

    console.log(`Start to save test data into ${csvName} file.`);
    let writeStream = await fs.createWriteStream(csvName);
    await csvStream.pipe(writeStream);
    return csvStream;
  },

  writeCSV: async function (stream, data) {
    await stream.write(data);
  },

  closeCSV: async function (stream) {
    await stream.end();
  }
};

module.exports = util;