const path = require('path');
const {DEV_CHROMIUM_PATH, NIGHTLY_BUILD_INFO} = require("../settings.js");
const util = require("../lib/Util.js");

const args = process.argv.slice(2);
const targetPlatform = util.getTargetPlatform();

(async () => {
  console.log(`>>> 2-Start install chromium build at ${(new Date()).toLocaleTimeString()}.`);
  util.uninstallChromium();

  if (args.length === 0) {
    const localPath = path.join(util.getChromiumPath(), NIGHTLY_BUILD_INFO[targetPlatform]["path"], util.getChromiumBuild());
    console.log(`2.2-Go to install nightly build (${localPath}).`);
    util.installChromium(localPath);
  } else {
    if (targetPlatform === "Linux") {
      console.log(`2.2(regression test)-Go to install dev build (${DEV_CHROMIUM_PATH}).`);
      util.installChromium(DEV_CHROMIUM_PATH);
    } else if (targetPlatform === "Windows") {
      console.log(`No need to do installation for testing dev build on Windows.`);
    }
  }
})().catch((err) => {
  throw err;
});