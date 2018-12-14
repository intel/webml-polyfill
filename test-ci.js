// const path = require('path');
const webdriver = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');
// const chromedriver = require('chromedriver');
// const until = webdriver.until;
const By = webdriver.By;
const options = new chrome.Options();
options.addArguments('no-sandbox');
const builder = new webdriver.Builder();
builder.forBrowser('chrome');
builder.setChromeOptions(options);
const driver = builder.build();
const csv = require('./node_modules/fast-csv');
const fs = require('fs');
const os = require('os');

let countPasses = 0;
let countFailures = 0;
let countPending = 0;
let flagPending = null;
let csvTitle = null;
let csvModule = null;
let csvName = null;
let csvPass = null;
let csvFail = null;
let csvNA = null;

(async function() {
  let baselinejson = JSON.parse(fs.readFileSync('./test/tools/CI/baseline/baseline.config.json'));
  let sys = os.type();
  let platform;
  if (sys == 'Linux') {
    platform = 'Linux';
  } else if (sys == 'Darwin') {
    platform = 'Mac';
  } else if (sys == 'Windows_NT') {
    platform = 'Windows';
  } else {
    let string = 'We do not support ' + sys + ' as run platform';
    throw new Error(string);
  }

  let getName = async function(element) {
    let Text = null;
    let length = 0;
    await element.findElement(By.xpath('./h2')).getText()
      .then(function(message) {
        length = message.length - 1;
        Text = message;
      });

    let arrayElement = await element.findElements(By.xpath('./h2/child::*'));
    for (let j = 1; j <= arrayElement.length; j++) {
      await arrayElement[j - 1].getText()
        .then(function(message) {
          length = length - message.length;
        });
    }

    if (flagPending) {
      return Text;
    } else {
      return Text.slice(0, length);
    }
  };

  let getError = async function(element) {
    let Text = await element.findElement(By.xpath('./pre[@class="error"]')).getText();
    return Text;
  };

  // let getCode = async function(element) {
  //   let Text = await element.findElement(By.xpath('./pre[last()]')).getText();
  //   return Text;
  // };
  let backendModel;
  let baseLineData = new Map();
  let lists = [
    'Feature',
    'Case Id',
    'Test Case',
    'Mac-WASM',
    'Mac-WebGL',
    'Android-WASM',
    'Android-WebGL',
    'Windows-WASM',
    'Windows-WebGL',
    'Linux-WASM',
    'Linux-WebGL',
  ];
  let checkStatus = async function(backendModel, results) {
    for (let i=0; i< lists.length; i++) {
      if (lists[i] == backendModel) {
        csv.fromPath('test/tools/CI/baseline/unitTestsBaseline.csv').on('data', function(data) {
          baseLineData.set(data[0] + data[2] + data[i]);
        }).on('end', function() {
          if (results['Pass'] != 1 && baseLineData.has(results['Feature'] + results['TestCase'] + 'Pass')) {
            let str = 'Feature: ' + results['Feature'] + ', TestCase: ' + results['TestCase'];
            failCaseList.push(str);
          }
        });
      }
    }
  };

  let getInfo = async function(element) {
    let array = await element.findElements(By.xpath('./ul/li[@class="test fail"]'));
    // just check fail test case, save time.

    for (let i = 1; i <= array.length; i++) {
      await array[i - 1].getAttribute('class')
        .then(function(message) {
          if (message == 'test pass pending') {
            flagPending = true;
          } else {
            flagPending = false;
          }
        });

      await getName(array[i - 1])
        .then(function(message) {
          csvName = message;
        });

      if (flagPending) {
        csvPass = null;
        csvFail = null;
        csvNA = '1';
        countPending++;
      } else {
        await getError(array[i - 1])
          .then(function(message) {
            csvPass = null;
            csvFail = '1';
            csvNA = null;
            countFailures++;
          }).catch(function(error) {
            csvPass = '1';
            csvFail = null;
            csvNA = null;
            countPasses++;
          });
      }

      if (csvModule == null) {
        csvModule = csvTitle;
      }

      let DataFormat = {
        Feature: csvTitle,
        CaseId: csvModule + '/' + i,
        TestCase: csvName,
        Pass: csvPass,
        Fail: csvFail,
        NA: csvNA,
      };
      await checkStatus(backendModel, DataFormat);
      csvName = null;
      csvPass = null;
      csvFail = null;
      csvNA = null;
    }
  };

  let grasp = async function() {
    let arrayTitles = await driver.findElements(By.xpath('//ul[@id="mocha-report"]/li[@class="suite"]'));
    for (let i = 1; i <= arrayTitles.length; i++) {
      await arrayTitles[i - 1].findElement(By.xpath('./h1/a')).getText()
        .then(function(message) {
          csvTitle = message;
          csvModule = null;
        });

      let arrayModule = await arrayTitles[i - 1].findElements(By.xpath('./ul/li[@class="suite"]'));
      for (let j = 1; j <= arrayModule.length; j++) {
        await arrayModule[j - 1].findElement(By.xpath('./h1/a')).getText()
          .then(function(message) {
            let array = message.split('#');
            csvModule = array[1];
          });
        await getInfo(arrayModule[j - 1]);
      }
      await getInfo(arrayTitles[i - 1]);
    }
  };

  let testResult = async function() {
    let backendModels = [
      'Mac-MPS',
      'Mac-BNNS',
      'Mac-WASM',
      'Mac-WebGL',
      'Android-NNAPI',
      'Android-WASM',
      'Android-WebGL',
      'Windows-clDNN',
      'Windows-WASM',
      'Windows-WebGL',
      'Linux-clDNN',
      'Linux-WASM',
      'Linux-WebGL',
    ];
    let backends = [
      'WASM',
      // 'WebGL'
    ];
    await driver.get('chrome://gpu');
    let vr = await driver.findElement(By.xpath('//*[@id="info-view-table"]/tbody/tr[2]/td[2]/span')).getText();
    await driver.sleep(1000);
    console.log('chrome version is :' + vr + '\n');
    for (let j of backends) {
      let totalResult;
      for (let i of backendModels) {
        if ((i.indexOf(platform) != -1) && (i.indexOf(j) != -1)) {
          backendModel = i;
          console.log('Begin test with : ' + i + ' backend.');
          totalResult = baselinejson[i];
          // let testlink = path.join('file:\/\/', __dirname, 'test', 'cts.html?backend=');
          let testlink = 'https://brucedai.github.io/nt/test/ci.html?backend=';
          await driver.get(testlink + j.toLowerCase());
          for (let t = 0; t <= 6; t++) {
            let time_begin = await driver.findElement(By.xpath('//ul[@id="mocha-stats"]/li[@class="duration"]//em')).getText();
            await driver.sleep(10000);
            let time_end = await driver.findElement(By.xpath('//ul[@id="mocha-stats"]/li[@class="duration"]//em')).getText();
            if (time_begin === time_end) {
              // add check, if pass/fail not match baseline will exit.
              let passResult = await driver.findElement(By.xpath('//*[@id="mocha-stats"]/li[2]/em')).getText();
              let failResult = await driver.findElement(By.xpath('//*[@id="mocha-stats"]/li[3]/em')).getText();
              if (totalResult.pass > passResult) {
                let str = 'Expect pass is :' + totalResult.pass + ' and actual result is : ' + passResult + ' will exit process !';
                throw new Error(str);
              }
              if (totalResult.fail < failResult) {
                let str = 'Expect fail is :' + totalResult.fail + ' and actual result is : ' + failResult + ' will exit process !';
                throw new Error(str);
              }
              break;
            };
          }
          countPasses = 0;
          countFailures = 0;
          countPending = 0;
          await grasp();
        }
      }
    }
    if (failCaseList.length > 0) {
      console.log('Test fail, below case get different result with expect data : ');
      for (let i = 0; i< failCaseList.length; i++ ) {
        console.log(failCaseList[i]);
      }
      await driver.quit();
      throw new Error('Test Fail');
    }
  };
  let failCaseList = [];
  await testResult();
  await driver.quit();
})().then(function() {
  console.log('Test completed!');
}).catch(function(err) {
  console.log(err);
  process.exit(1);
});
