const fs = require('fs');
const path = require('path');
const csv = require('csvtojson');
const jsdom = require("jsdom");
const {chromium} = require('playwright');
const settings = require("../settings.js");
const util = require("../lib/Util.js");

const targetPlatform = util.getTargetPlatform();
const args = process.argv.slice(2);
const reportHTML = path.join(process.cwd(), "output", "report", 'check-report.html');
let findings = [];

const showReport = async (reportURL) => {
  let chromiumPath;
  if (targetPlatform === "Linux") {
    chromiumPath = '/usr/bin/chromium-browser-unstable';
  } else if (targetPlatform === "Windows") {
    chromiumPath = settings.DEV_CHROMIUM_PATH;
  }

  const browser = await chromium.launch({headless: false, executablePath: chromiumPath, args: ['--no-sandbox']});
  const context = await browser.newContext({ignoreHTTPSErrors: true, viewport: null});
  const page = await context.newPage();
  const contentHtml = fs.readFileSync(reportURL, 'utf8');
  await page.setContent(contentHtml);
};

const createDivEle = (dom, divId) => {
  let divEle = dom.window.document.createElement('div');
  divEle.id = divId;
  return divEle;
};

const createTableEle = (dom, backend) => {
  let tableEle = dom.window.document.createElement('table');
  tableEle.id = `${backend}Table`;
  tableEle.border = "1";
  return tableEle;
};

const createTheadEle = (dom, backend) => {
  let theadEle = dom.window.document.createElement('thead');
  let trEle = dom.window.document.createElement('tr');
  let thEle = dom.window.document.createElement('th');
  thEle.innerHTML = "CATEGORY";
  trEle.appendChild(thEle);

  thEle = dom.window.document.createElement('th');
  thEle.innerHTML = "MODEL";
  trEle.appendChild(thEle);

  thEle = dom.window.document.createElement('th');
  thEle.innerHTML = `baseline ${backend}`;
  trEle.appendChild(thEle);

  thEle = dom.window.document.createElement('th');
  thEle.innerHTML = `${backend}`;
  trEle.appendChild(thEle);

  thEle = dom.window.document.createElement('th');
  thEle.innerHTML = `Speedup`;
  trEle.appendChild(thEle);
  theadEle.appendChild(trEle);
  return theadEle;
};

const createTbodyTrEle = (dom, category, model, backend, baselineData, data) => {
  let trEle = dom.window.document.createElement('tr');
  let tdEle = dom.window.document.createElement('td');
  tdEle.innerHTML = category;
  trEle.appendChild(tdEle);

  tdEle = dom.window.document.createElement('td');
  tdEle.innerHTML = model;
  trEle.appendChild(tdEle);

  tdEle = dom.window.document.createElement('td');
  if (baselineData === '') {
    tdEle.innerHTML = 'NA';
  } else {
    tdEle.innerHTML = baselineData.split('/').join('<br>');
  }

  trEle.appendChild(tdEle);

  tdEle = dom.window.document.createElement('td');
  if (data === '') {
    tdEle.innerHTML = 'NA';
  } else {
    tdEle.innerHTML = data.split('/').join('<br>');
  }
  trEle.appendChild(tdEle);

  tdEle = dom.window.document.createElement('td');
  if (baselineData === '' || data === '') {
    tdEle.innerHTML = 'NA';
  } else {
    let result = subCompare(data, baselineData);
    tdEle.innerHTML = result;
    if (result.includes('FAIL')) {
      findings.push(`${category} ${model} test by ${backend} has regression "${result}", please re-check it.`);
    }
  }
  trEle.appendChild(tdEle);

  return trEle;
};

const getBaselineRow = (rowDicList, category, model) => {
  for (let row of rowDicList) {
    if (row["CATEGORY"] === category && row["MODEL"] === model) {
      return row;
    }
  }

  return null;
};

const scoreCompare = (compare, baseline) => {
  let status = Number(compare) / Number(baseline) * 100;
  return `${status.toFixed(2)}%`;
};

const labelCompare = (compare, baseline) => {
  let status = 'OK';
  let [compareLabel, compareProbability] = compare.split(':');
  let [baselineLabel, baselineProbability] = baseline.split(':');

  if (compareLabel !== baselineLabel) {
    status = 'FAIL by Top1 lable';
  } else {
    const np1 = Number(compareProbability.split('%')[0]);
    const np2 = Number(baselineProbability.split('%')[0]);
    if (np1 < np2) {
      status = 'FAIL by downgrade of Top1 probability';
    }
  }

  return status;
};

const subCompare = (compare, baseline) => {
  let [compareScore, compareTop1] = compare.split('/').slice(0,2);
  let [baselineScore, baselineTop1] = baseline.split('/').slice(0,2);
  let scoreStatus = scoreCompare(compareScore.split('+')[0], baselineScore.split('+')[0]);
  let labelStatus = labelCompare(compareTop1, baselineTop1);
  return labelStatus !== 'OK' ? labelStatus : scoreStatus;
};

(async () => {
  let compareFile;
  let baselineFile;

  if (args.length === 1) {
    compareFile = args[0];
    baselineFile = path.join("output", "report", "baseline", `baseline-${targetPlatform.toLocaleLowerCase()}.csv`);
    if (!fs.existsSync(baselineFile)) {
      baselineFile = path.join("baseline", `baseline-${targetPlatform.toLocaleLowerCase()}.csv`);
      if (!fs.existsSync(baselineFile)) {
        throw new Error("baseline file doesn't exist, please check it.");
      }
    }
  } else if (args.length === 2) {
    compareFile = args[0];
    baselineFile = args[1];
  } else {
    baselineFile = path.join("output", "report", "baseline", `baseline-${targetPlatform.toLocaleLowerCase()}.csv`);
    if (!fs.existsSync(baselineFile)) {
      baselineFile = path.join("baseline", `baseline-${targetPlatform.toLocaleLowerCase()}.csv`);
      if (!fs.existsSync(baselineFile)) {
        throw new Error("baseline file doesn't exist, please check it.");
      }
    }
    compareFile  = path.join("output", "report", "dev", `result-${targetPlatform.toLocaleLowerCase()}.csv`);
  }

  const baselineConfig = await csv().fromFile(baselineFile);
  const compareConfig = await csv().fromFile(compareFile);
  const compareRowLen = compareConfig.length;
  const backendList = Object.keys(compareConfig[0]).slice(2);

  let htmlSource = fs.readFileSync(path.join('static', 'template', 'check-report-template.html'), "utf8");
  let dom = new jsdom.JSDOM(htmlSource);
  let h2Ele = dom.window.document.createElement('h2');
  h2Ele.innerHTML = `Check Report on ${targetPlatform}`;
  let hrEle = dom.window.document.createElement('hr');
  let h3Ele = dom.window.document.createElement('h3');
  h3Ele.innerHTML = 'Detials:';

  let detailDiv = createDivEle(dom, 'detial');

  if (args.length === 0) {
    let baselineInfoPEle = dom.window.document.createElement('p');
    let baselineCommitId = util.getLatestCommit();
    let baselineBuildURL = settings.NIGHTLY_BUILD_URL + baselineCommitId + '/';
    baselineInfoPEle.innerHTML = `Baseline commit: <a href="${baselineBuildURL}" target="_blank">${baselineCommitId}</a>`;
    detailDiv.appendChild(baselineInfoPEle);
  }

  for (let backend of backendList) {
    let tableEle = createTableEle(dom, backend);
    let theadEle = createTheadEle(dom, backend);
    let tbodyEle = dom.window.document.createElement('tbody');
    for (let rowIndex = 0; rowIndex <  compareRowLen; rowIndex++) {
      let compareRow = compareConfig[rowIndex];
      let category = compareRow["CATEGORY"];
      let model = compareRow["MODEL"];
      let baselineRow = getBaselineRow(baselineConfig, category, model);
      let trEle = createTbodyTrEle(dom, category, model, backend, baselineRow[backend], compareRow[backend]);
      tbodyEle.appendChild(trEle);
    }
    tableEle.appendChild(theadEle);
    tableEle.appendChild(tbodyEle);
    detailDiv.appendChild(tableEle);
    detailDiv.appendChild(dom.window.document.createElement('br'));
  }

  dom.window.document.body.appendChild(h2Ele);
  dom.window.document.body.appendChild(hrEle);
  dom.window.document.body.appendChild(h3Ele);
  dom.window.document.body.appendChild(detailDiv);

  let summaryDiv = createDivEle(dom, 'summary');
  let summaryH3Ele = dom.window.document.createElement('h3');
  summaryH3Ele.innerHTML = "Summary:";
  summaryDiv.appendChild(summaryH3Ele);

  if (findings.length !== 0) {
    let index = 1;
    for (let f of findings) {
      let pEle = dom.window.document.createElement('p');
      pEle.innerHTML = `${index}. ${f}`;
      summaryDiv.appendChild(pEle);
      index++;
    }
  } else {
    let pEle = dom.window.document.createElement('p');
    pEle.innerHTML = `PASS without any obvious regression.`;
    summaryDiv.appendChild(pEle);
  }

  dom.window.document.body.appendChild(summaryDiv);

  fs.writeFileSync(reportHTML, dom.serialize());
  console.log(`Saved compare report as ${reportHTML}.`);

  await showReport(reportHTML);
})();