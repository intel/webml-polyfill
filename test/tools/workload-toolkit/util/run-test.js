const { chromium } = require('playwright');
const path = require('path');
const sleep = require('sleep-promise');
const settings = require("../settings.js");
const util = require("../lib/Util.js");

const runCheckingFlag = process.argv.slice(2)[0] === 'regression-test';
const targetPlatform = util.getTargetPlatform();
let unsuccessTests = [];

const launchWorkloadPage = async (args = ['--no-sandbox']) => {
  let chromiumPath;
  if (targetPlatform == "Linux") {
    chromiumPath = '/usr/bin/chromium-browser-unstable';
  } else if (targetPlatform == "Windows") {
    chromiumPath = path.join(util.getChromiumPath(), settings.NIGHTLY_BUILD_INFO[targetPlatform]["path"], 'Chrome-bin', 'chrome.exe');
    if (runCheckingFlag) {
      chromiumPath = settings.DEV_CHROMIUM_PATH;
    }
  }

  const browser = await chromium.launch({headless: false, executablePath: chromiumPath, args: args});
  const context = await browser.newContext({ignoreHTTPSErrors: true});
  const page = await context.newPage();
  await page.goto(settings.WORKLOAD_URL);
  return [browser, page];
};

const getCategoryList = async () => {
  if (settings.REGRESSION_FLAG) {
    return Object.keys(settings.REGRESSION_TEST);
  } else {
    let categoryList = [];
    const [browser, page] = await launchWorkloadPage();
    // category select
    const catgOptSelect = await page.$$('select#categoryselect option');
    for (let opt of catgOptSelect) {
      let catg = await opt.evaluate(element => element.textContent);
      categoryList.push(catg);
    }
    await browser.close();
    return categoryList;
  }
};

const getModelList = async (category) => {
  if (settings.REGRESSION_FLAG) {
    return settings.REGRESSION_TEST[category];
  } else {
    let modelList = [];
    const [browser, page] = await launchWorkloadPage();
    // category select
    const catgSelect = await page.$('select#categoryselect');
    await catgSelect.type(category);
    const modelOptSelect = await page.$$('select#modelselect1 option');
    for (let opt of modelOptSelect) {
      let model = await opt.evaluate(element => element.textContent);
      modelList.push(model);
    }
    await browser.close();
    return modelList;
  }
};

const getSkipStatus = (category, model, backend) => {
  const filter1 = settings.CATEGORY_FILTER;
  const filter2 = settings.MODEL_FILTER;
  let status = false;

  // check category supported status by backend
  if (filter1.hasOwnProperty(category)) {
    status = filter1[category].includes(backend);
  }

  // check model supported status by backend
  return status || filter2[backend].includes(model);
};

const executeWorkloadTest = async (category, model, config) => {
  const [browser, page] = await launchWorkloadPage(config.args);
  page.setDefaultTimeout((settings.ITERATIONS + 1) * 20000);

  // category select
  const catgSelect = await page.$('select#categoryselect');
  await catgSelect.type(category);

  const modelSelect = await page.$('select#modelselect1');
  await modelSelect.type(model);

  const backendSelect = await page.$('select#webnnbackend');
  await backendSelect.type(config.backend);

  const preferSelect = await page.$('select#preferselect');
  await preferSelect.type(config.prefer);

  await page.fill('#iterations', settings.ITERATIONS.toString());

  await page.click('#runbutton');

  let score;
  let note = '';

  try {
    if (category === 'Image Classification') {
      await page.waitForSelector("#imageclassificationlabels").then(async () => {
        for (let i = 0; i < 3; i++) {
          const labelEle = await page.$('#label'+i);
          const label = await labelEle.evaluate(element => element.textContent);
          const probEle = await page.$('#prob'+i);
          const prob = await probEle.evaluate(element => element.textContent);
          note += '/' + label + ':' + prob;
        }
      });
    }
    await page.waitForSelector("em").then(async () => {
      const resultElement = await page.$('em');
      score = await resultElement.evaluate(element => element.textContent);
    });
  } catch (e) {
    unsuccessTests.push(`${config.backend} + ${config.prefer} + ${model}`);
    console.log(`${config.backend} + ${config.prefer} + ${model}: ${e}`);
  }

  await browser.close();
  return score + note;
};

(async () => {
  console.log(`>>> 3-Start test at ${(new Date()).toLocaleTimeString()}`);
  let csvStream = await util.openCSV(runCheckingFlag);
  const categoryList =  await getCategoryList();
  const realTargetBackend = util.getRealTargetBackend();

  for (let category of categoryList) {
    console.log(`###### Start test ${category} workload at ${(new Date()).toLocaleTimeString()} ######`);
    let modelList = await getModelList(category);
    for (let model of modelList) {
      console.log(`$$$$$$ ${model} $$$$$$`);
        let content = {
          category: category,
          model: model,
        };
        for (let backend of realTargetBackend) {
          const sub_config = settings.BACKEND_CONFIG[backend];
          let is_skip = getSkipStatus(category, model, backend);
          if (is_skip) {
            console.log(`Skip test ${sub_config.backend} + ${sub_config.prefer} + ${category} / ${model}`);
            continue;
          }
          content[backend.toLocaleLowerCase()] = await executeWorkloadTest(category, model, sub_config);
          if (content[backend.toLocaleLowerCase()] === 'undefined') {
            console.log(`${backend} --- unsuccessfull test`);
          } else {
            console.log(`${backend} --- ${content[backend.toLocaleLowerCase()]}`);
          }
          // sleep 5 minutes to get stable data for next test
          await sleep(300000);
        }
        await util.writeCSV(csvStream, content);
    }
  }

  await util.closeCSV(csvStream);

  if (unsuccessTests.length > 0) {
    console.log(`>>> Following are unsuccessfull tests, please manually re-check.`);
    for (let msg of unsuccessTests) {
      console.log(msg);
    }
  }
  console.log(`>>> 3-Completed test at ${(new Date()).toLocaleTimeString()}`);
})();
