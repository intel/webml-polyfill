const Builder = require("../node_modules/selenium-webdriver").Builder;
const By = require("../node_modules/selenium-webdriver").By;
const until = require("../node_modules/selenium-webdriver").until;
const Chrome = require("../node_modules/selenium-webdriver/chrome");
const csv = require("../node_modules/fast-csv");
const execSync = require("child_process").execSync;
const fs = require("fs");
const os = require("os");
require("chromedriver");

var outputPath = "./output";
if (!fs.existsSync(outputPath)) {
    fs.mkdirSync(outputPath);
}

var htmlPath = outputPath + "/report-check-result.html";

var htmlStream = fs.createWriteStream(htmlPath, {flags: "a"});

var remoteURL, driver, backendModel, chromeOption, command, androidSN, adbPath, htmlPath;
var backendModels = [
    "Mac-MPS",
    "Mac-BNNS",
    "Mac-WASM",
    "Mac-WebGL",
    "Android-NNAPI",
    "Android-WASM",
    "Android-WebGL",
    "Windows-clDNN",
    "Windows-WASM",
    "Windows-WebGL",
    "Linux-clDNN",
    "Linux-WASM",
    "Linux-WebGL"
];

var RCjson = JSON.parse(fs.readFileSync("./config.json"));
var testPlatform = RCjson.platform;
var chromiumPath = RCjson.chromiumPath;

var baselinejson = JSON.parse(fs.readFileSync("./baseline/baseline.config.json"));
var versionChromium = baselinejson.Version.chromium;
var versionPolyfill = baselinejson.Version.polyfill;

/**
 * baseLineData = {
 *     key: {
 *         "Feature": value,
 *         "CaseId": value,
 *         "TestCase": value,
 *         "Mac-MPS": value,
 *         "Mac-BNNS": value,
 *         "Mac-WASM": value,
 *         "Mac-WebGL": value,
 *         "Android-NNAPI": value,
 *         "Android-WASM": value,
 *         "Android-WebGL": value,
 *         "Windows-clDNN": value,
 *         "Windows-WASM": value,
 *         "Windows-WebGL": value,
 *         "Linux-clDNN": value,
 *         "Linux-WASM": value,
 *         "Linux-WebGL": value
 *     }
 * }
 */
var baseLineData = new Map();
/**
 * pageData = {
 *     backend: {
 *         "pass2fail": [value1, value2, value3],
 *         "fail2pass": [value1, value2, value3]
 *     }
 * }
 */
var pageData = new Map();
/**
 * pageDataTotal = {
 *     backend: {
 *         "Baseline": [total, pass, fail, block, passrate],
 *         "grasp": [total, pass, fail, block, passrate]
 *     }
 * }
 */
var pageDataTotal = new Map();
for (let i = 0; i < backendModels.length; i++) {
    pageData.set(backendModels[i], new Map([["pass2fail", new Array()], ["fail2pass", new Array()]]));
    pageDataTotal.set(backendModels[i], new Map([["Baseline", new Array(
        baselinejson[backendModels[i]]["total"],
        baselinejson[backendModels[i]]["pass"],
        baselinejson[backendModels[i]]["fail"],
        baselinejson[backendModels[i]]["block"],
        Math.round((baselinejson[backendModels[i]]["pass"] / baselinejson[backendModels[i]]["total"]) * 100).toString() + "%"
    )], ["grasp", new Array()]]));
}

var testBackends = new Array();
var crashData = new Array();
/**
 * newTestCaseData = {
 *     "caseCount": value,
 *     "backends": {
 *         backend1: true,
 *         backend2: true
 *     },
 *     testCase: {
 *         "title": title,
 *         "caseID": caseID,
 *         "backend": {
 *             backend1: caseStatus1,
 *             backend2: caseStatus2,
 *             backend3: caseStatus3
 *         }
 *     }
 * }
 */
var newTestCaseData = new Map();
newTestCaseData.set("caseCount", 0);
newTestCaseData.set("backends", new Map());

/**
 * graspDataSummary = [
 *     total: value,
 *     pass: value,
 *     fail: value,
 *     block: value
 * ]
 */
var graspDataSummary = new Array();
/**
 * baseLineDataSummary = [
 *     model-name1: count1,
 *     model-name2: count2,
 *     model-name3: count3
 * ]
 */
var baseLineDataSummary = new Map();

csv.fromPath("./baseline/unitTestsBaseline.csv").on("data", function(data){
    baseLineData.set(data[0] + "-" + data[1], new Map(
        [
            ["Feature", data[0]],
            ["CaseId", data[1]],
            ["TestCase", data[2]],
            ["Mac-MPS", data[3]],
            ["Mac-BNNS", data[4]],
            ["Mac-WASM", data[5]],
            ["Mac-WebGL", data[6]],
            ["Android-NNAPI", data[7]],
            ["Android-WASM", data[8]],
            ["Android-WebGL", data[9]],
            ["Windows-clDNN", data[10]],
            ["Windows-WASM", data[11]],
            ["Windows-WebGL", data[12]],
            ["Linux-clDNN", data[13]],
            ["Linux-WASM", data[14]],
            ["Linux-WebGL", data[15]]
        ]
    ));

    let newData = data[1].split("/")[0];
    let dataArray = data[1].split("/");
    if (dataArray.length > 2) {
        for (let dataCount = 1; dataCount < dataArray.length - 1; dataCount++) {
            newData = newData + "/" + dataArray[dataCount];
        }
    }

    baseLineData.set(data[0] + "-" + newData + "-" + data[2], new Map(
        [
            ["Feature", data[0]],
            ["CaseId", newData],
            ["TestCase", data[2]],
            ["Mac-MPS", data[3]],
            ["Mac-BNNS", data[4]],
            ["Mac-WASM", data[5]],
            ["Mac-WebGL", data[6]],
            ["Android-NNAPI", data[7]],
            ["Android-WASM", data[8]],
            ["Android-WebGL", data[9]],
            ["Windows-clDNN", data[10]],
            ["Windows-WASM", data[11]],
            ["Windows-WebGL", data[12]],
            ["Linux-clDNN", data[13]],
            ["Linux-WASM", data[14]],
            ["Linux-WebGL", data[15]]
        ]
    ));

    if (baseLineDataSummary.has(data[0] + "-" + newData)) {
        baseLineDataSummary.set(data[0] + "-" + newData, baseLineDataSummary.get(data[0] + "-" + newData) + 1);
    } else {
        baseLineDataSummary.set(data[0] + "-" + newData, 1);
    }
}).on("end", function() {
    for (let key of baseLineData.keys()) {
        RClog("debug", "key: " + key);
    }

    for (let key of baseLineDataSummary.keys()) {
        RClog("debug", "key: " + key);
        RClog("debug", "value: " + baseLineDataSummary.get(key));
    }
});

var continueFlag = false;
var debugFlag = false;
var timeFlag = false;
function RClog (target, message) {
    if (target == "console") {
        console.log("RC -- " + message);
    } else if (target == "debug") {
        if (debugFlag) console.log("RC -- " + message);
    } else if (target == "time") {
        // For performance testing
        if (timeFlag) {
            console.log("RC -- running time: " + process.uptime() + "s");
        }
    } else {
        throw new Error("Not support target '" + target + "'");
    }
}

RClog("console", "checking runtime environment....");
RClog("time", "mark");

if (testPlatform == "Android") {
    RClog("console", "runtime environment: android");

    var sys = os.type();

    if (sys == "Linux") {
        adbPath = "./lib/adb-tool/Linux/adb";

        try {
            command = "killall adb";
            execSync(command, {encoding: "UTF-8", stdio: "pipe"});
        } catch(e) {
            if (e.message.search("no process found") == -1) {
                throw e;
            }
        }
    } else if (sys == "Darwin") {
        adbPath = "./lib/adb-tool/Mac/adb";

        try {
            command = "killall adb";
            execSync(command, {encoding: "UTF-8", stdio: "pipe"});
        } catch(e) {
            if (e.message.search("No matching processes") == -1) {
                throw e;
            }
        }
    } else if (sys == "Windows_NT") {
        adbPath = ".\\lib\\adb-tool\\Windows\\adb";

        try {
            command = "taskkill /im adb.exe /f";
            execSync(command, {encoding: "UTF-8", stdio: "pipe"});
        } catch(e) {
            if (e.message.search("not found") == -1) {
                throw e;
            }
        }
    }

    command = adbPath + " start-server";
    execSync(command, {encoding: "UTF-8", stdio: "pipe"});

    try {
        command = adbPath + " devices";
        let log = execSync(command, {encoding: "UTF-8", stdio: "pipe"}).split(/\s+/);

        let array = new Array();
        for (let i = 0; i < log.length; i++) {
            if (log[i] == "device") array.push(log[i - 1]);
        }

        if (array.length == 0) {
            throw new Error("no android device");
        } else if (array.length > 1) {
            androidSN = array[0];
            RClog("console", "more android devices, using the first one: " + array[0]);
        } else {
            androidSN = array[0];
            RClog("console", "android device: " + array[0]);
        }
    } catch(e) {
        throw e;
    }

    try {
        command = adbPath + " -s " + androidSN + " shell pm list packages | grep org.chromium.chrome";
        execSync(command, {encoding: "UTF-8", stdio: "pipe"});
        RClog("console", "chromium to be tested is installed correctly");
    } catch(e) {
        throw new Error("chromium to be tested is not installed correctly");
    }

    command = adbPath + " -s " + androidSN + " shell am force-stop org.chromium.chrome";
    execSync(command, {encoding: "UTF-8", stdio: "pipe"});
} else if (testPlatform == "Linux") {
    RClog("console", "runtime environment: Linux");

    if (fs.existsSync(chromiumPath)) {
        RClog("console", "chromium to be tested is installed correctly");
    } else {
        throw new Error("chromium to be tested is not installed correctly");
    }

    try {
        command = "ps aux | grep chrome";
        let Lines = execSync(command, {encoding: "UTF-8", stdio: "pipe"}).trim().split("\n");
        for (let line of Lines) {
            let infos = line.trim().split(/\s+/);
            if (infos[10] == "/opt/chromium.org/chromium-unstable/chrome") {
                command = "kill " + infos[1];
                execSync(command, {encoding: "UTF-8", stdio: "pipe"});
            }
        }
    } catch(e) {
        if (e.message.search("No such process") == -1) {
            throw e;
        }
    }
} else if (testPlatform == "Mac") {
    RClog("console", "runtime environment: Mac");

    if (fs.existsSync(chromiumPath)) {
        RClog("console", "chromium to be tested is installed correctly");
    } else {
        throw new Error("chromium to be tested is not installed correctly");
    }

    try {
        command = "killall Chromium";
        execSync(command, {encoding: "UTF-8", stdio: "pipe"});
    } catch(e) {
        if (e.message.search("No matching processes") == -1) {
            throw e;
        }
    }
} else if (testPlatform == "Windows") {
    RClog("console", "runtime environment: Windows");

    if (fs.existsSync(chromiumPath)) {
        RClog("console", "chromium to be tested is installed correctly");
    } else {
        throw new Error("chromium to be tested is not installed correctly");
    }

    try {
        command = "wmic process where name='chrome.exe' get processid";
        let idLines = execSync(command, {encoding: "UTF-8", stdio: "pipe"}).trim().split("\n");
        for (let idLine of idLines) {
            idLine = idLine.trim();
            if (idLine !== "ProcessId") {
                command = "wmic process where processid='" + idLine + "' get executablepath";
                let pathLine = execSync(command, {encoding: "UTF-8", stdio: "pipe"}).trim().split("\n");
                if (pathLine[1] == chromiumPath) {
                    command = "wmic process where processid='" + idLine + "' delete";
                    execSync(command, {encoding: "UTF-8", stdio: "pipe"});
                }
            }
        }
    } catch(e) {
        if (e.message.search("not found") == -1) {
            throw e;
        }
    }
}

RClog("time", "mark");

var numberPasstoFail = 0;
var numberFailtoPass = 0;
var numberTotal = 0;
var matchFlag = null;

(async function() {
    RClog("console", "checking chromium code is start");

    var getCaseStatus = async function(element) {
        return element.getAttribute("class").then(function(message) {
            let graspCaseStatus = null;
            if (message == "test pass pending") {
                graspCaseStatus = "N/A";
            } else if (message == "test pass fast" || message == "test pass slow" || message == "test pass medium") {
                graspCaseStatus = "Pass";
            } else if (message == "test fail") {
                graspCaseStatus = "Fail";
            } else {
                throw new Error("not support case status");
            }

            return graspCaseStatus;
        });
    }

    var getCaseName = async function(element) {
        let Text = null;
        let length = 0;
        await element.findElement(By.xpath("./h2")).getText().then(function(message) {
            length = message.length - 1;
            Text = message;
        });

        let arrayElement = await element.findElements(By.xpath("./h2/child::*"));
        for (let j = 1; j <= arrayElement.length; j++) {
            await arrayElement[j - 1].getText().then(function(message) {
                length = length - message.length;
            });
        }

        return Text.slice(0, length);
    }

    var checkResult = async function(element, count, title, module, flag) {
        let caseStatus = await getCaseStatus(element);
        let caseName, key;

        if (flag) {
            key = title + "-" + module + "/" + count;
            caseName = baseLineData.get(key).get("TestCase");
        } else {
            caseName = await getCaseName(element);
            key = title + "-" + module + "-" + caseName;
        }

        if (baseLineData.has(key)) {
            graspDataSummary["total"] = graspDataSummary["total"] + 1;
            if (caseStatus == "Pass") {
                graspDataSummary["pass"] = graspDataSummary["pass"] + 1;
            } else if (caseStatus == "Fail") {
                graspDataSummary["fail"] = graspDataSummary["fail"] + 1;
            } else if (caseStatus == "N/A") {
                graspDataSummary["block"] = graspDataSummary["block"] + 1;
            }

            let baseLineStatus = baseLineData.get(key).get(backendModel);
            if (caseStatus !== baseLineStatus) {
                if (baseLineStatus == "Pass" && caseStatus == "Fail") {
                    pageData.get(backendModel).get("pass2fail").push([title, module + "-" + caseName]);
                } else {
                    pageData.get(backendModel).get("fail2pass").push([title, module + "-" + caseName]);
                }

                RClog("console", title + "-" + module + "-" + caseName);
                RClog("console", "baseLineStatus: " + baseLineStatus + " - caseStatus: " + caseStatus);
            }
        } else {
            RClog("console", "no match test case: " + title + "-" + module + "-" + caseName);

            if (matchFlag == "macth" || matchFlag == "delete") {
                throw new Error("no match test case: " + title + "-" + module + "-" + caseName);
            } else {
                if (!newTestCaseData.get("backends").has(backendModel)) {
                    newTestCaseData.get("backends").set(backendModel, true);
                }

                if (!newTestCaseData.has(title + "-" + module + "-" + caseName)) {
                    newTestCaseData.set(title + "-" + module + "-" + caseName, new Map());
                    newTestCaseData.get(title + "-" + module + "-" + caseName).set("title", title);
                    newTestCaseData.get(title + "-" + module + "-" + caseName).set("caseID", module + "-" + caseName);
                    newTestCaseData.get(title + "-" + module + "-" + caseName).set("backend", new Map());
                    newTestCaseData.set("caseCount", newTestCaseData.get("caseCount") + 1);
                }

                newTestCaseData.get(title + "-" + module + "-" + caseName).get("backend").set(backendModel, caseStatus);
            }
        }
    }

    var graspResult = async function() {
        let actions = 0;
        let actionCount = 0;
        let graspTotal, graspPass, graspFail;

        await driver.findElement(By.xpath("//ul[@id='mocha-stats']/li[@class='passes']//em")).getText().then(function(message) {
            RClog("debug", "passes: " + message);
            graspPass = message >> 0;
        });

        await driver.findElement(By.xpath("//ul[@id='mocha-stats']/li[@class='failures']//em")).getText().then(function(message) {
            RClog("debug", "failures: " + message);
            graspFail = message >> 0;
        });

        graspTotal = graspPass + graspFail;
        if (graspTotal == baselinejson[backendModel]["total"]) {
            matchFlag = "macth";
            RClog("console", "match base line data: matching mode");
        } else {
            if (graspTotal > baselinejson[backendModel]["total"]) {
                matchFlag = "add";
            } else if (graspTotal < baselinejson[backendModel]["total"]) {
                matchFlag = "delete";
            }

            RClog("console", "not match base line data: compatibility mode");
            RClog("console", "will too slow, because grasping more information");
        }

        await driver.findElements(By.xpath("//ul[@id='mocha-report']/li[@class='suite']")).then(function(arrayTitles) {
            for (let i = 0; i < arrayTitles.length; i++) {
                arrayTitles[i].findElement(By.xpath("./h1/a")).getAttribute("textContent").then(function(message) {
                    let title = message;

                    arrayTitles[i].findElements(By.xpath("./ul/li[@class='suite']")).then(function(arrayModules) {
                        if (arrayModules.length === 0) {
                            let module = title;

                            arrayTitles[i].findElements(By.xpath("./ul/li[@class='test pass fast' or " +
                                                                 "@class='test pass slow' or " +
                                                                 "@class='test fail' or " +
                                                                 "@class='test pass pending' or " +
                                                                 "@class='test pass medium']")).then(async function(arrayCase) {
                                RClog("debug", "title: " + title + "    module: " + module + "    case: " + arrayCase.length);

                                let modeFlag;
                                if (baseLineDataSummary.get(title + "-" + module) == arrayCase.length) {
                                    modeFlag = true;
                                    RClog("debug", title + "-" + module + ": match, fast search mode");
                                } else {
                                    modeFlag = false;
                                    RClog("console", title + "-" + module + ": not match, carefully search mode");
                                }

                                for (let k = 0; k < arrayCase.length; k++) {
                                    await checkResult(arrayCase[k], k + 1, title, module, modeFlag).then(function() {
                                        actions = actions + 1;
                                    });
                                }
                            });
                        } else {
                            for (let j = 0; j < arrayModules.length; j++) {
                                arrayModules[j].findElement(By.xpath("./h1/a")).getAttribute("textContent").then(function(message) {
                                    let module = message.split("#")[1];

                                    arrayModules[j].findElements(By.xpath("./ul/li[@class='test pass fast' or " +
                                                                          "@class='test pass slow' or " +
                                                                          "@class='test fail' or " +
                                                                          "@class='test pass pending' or " +
                                                                          "@class='test pass medium']")).then(async function(arrayCase) {
                                        RClog("debug", "title: " + title + "    module: " + module + "    case: " + arrayCase.length);

                                        let modeFlag;
                                        if (baseLineDataSummary.get(title + "-" + module) == arrayCase.length) {
                                            modeFlag = true;
                                            RClog("debug", title + "-" + module + ": match, fast search mode");
                                        } else {
                                            modeFlag = false;
                                            RClog("console", title + "-" + module + ": not match, carefully search mode");
                                        }

                                        for (let k = 0; k < arrayCase.length; k++) {
                                            await checkResult(arrayCase[k], k + 1, title, module, modeFlag).then(function() {
                                                actions = actions + 1;
                                            });
                                        }
                                    });
                                });
                            }
                        }
                    });
                });
            }
        });

        await driver.wait(function() {
            if (actionCount != actions) {
                actionCount = actions;
                RClog("debug", graspTotal + " : " + actionCount);
            }

            return (actions == graspTotal);
        }, 5000000).then(function() {
            RClog("console", "grasp all test case: " + graspTotal);
        }).catch(function() {
            RClog("console", "total: " + graspTotal + " grasp: " + actionCount);
            throw new Error("failed to grasp all test result");
        });
    }

    var createHtmlHead = function() {
        let htmlDataHead = "\
  <head>\n\
    <meta charset='utf-8'>\n\
    <title>PR Submission Checking Summary</title>\n\
    <style>\n\
    h2 {text-align:center}\n\
    .container {margin: 20px 20px}\n\
    .suggest {color:green}\n\
    .notsuggest {color:red}\n\
    .tab-menu {margin: 10px 0px -10px 0px;}\n\
    .tab-menu ul {height:30px;border-bottom:1px solid gray;list-style:none;padding-left:0;}\n\
    .tab-menu ul li {float:left;width:150px;margin-right:3px;color:#000;border:solid 1px gray;border-bottom:none; text-align:center;line-height:30px;}\n\
    .tab-menu ul li.active {background-color: #007bc7;color: #fff;}\n\
    .tab-menu ul li:hover {cursor: pointer;}\n\
    .tab-box div {display:none;}\n\
    .tab-box div.active {display:block;}\n\
    .tab-box div div.NewTestCase {display:block;}\n\
    table {border: 1px solid #ddd; border-spacing:0;}\n\
    table tr th {border: 1px solid #000;background-color: #B0C4DE;}\n\
    table tr td {border: 1px solid #ddd}\n\
    table tr.fail2pass {display:none;}\n\
    .warnning {color:red}\n\
    .pass {color:green}\n\
    .fail {color:red}\n\
    </style>\n\
    <script>\n\
      function tab1_click() {\n\
        document.getElementById('tab_menu2').classList.remove('active');\n\
        document.getElementById('tab_menu1').classList.add('active');\n\
        for ( let node of document.getElementsByClassName('pass2fail') ) {\n\
          node.style.display = 'table-row';\n\
        }\n\
        for ( let node of document.getElementsByClassName('fail2pass') ) {\n\
          node.style.display = 'none';\n\
        }\n\
      }\n\
      function tab2_click() {\n\
        document.getElementById('tab_menu1').classList.remove('active');\n\
        document.getElementById('tab_menu2').classList.add('active');\n\
        for ( let node of document.getElementsByClassName('pass2fail') ) {\n\
          node.style.display = 'none';\n\
        }\n\
        for ( let node of document.getElementsByClassName('fail2pass') ) {\n\
          node.style.display = 'table-row';\n\
        }\n\
      }\n\
    </script>\n\
  </head>\n";

        htmlStream.write(htmlDataHead);
    }

    var createHtmlBodyContainerVersion = function(space) {
        htmlStream.write(space + "<div>\n");
        htmlStream.write(space + "  <h2>PR Submission Checking Summary</h2>\n");
        htmlStream.write(space + "  <hr />\n");
        htmlStream.write(space + "  <h3>Baseline Information:</h3>\n");
        htmlStream.write(space + "    <div>Chromium version: " + versionChromium + "</div>\n");
        htmlStream.write(space + "    <div>Webml-polyfill version: " + versionPolyfill + "</div>\n");
        htmlStream.write(space + "</div>\n");

        htmlStream.write(space + "<hr />\n");
    }

    var createHtmlBodyContainerCrash = function(space) {
        if (crashData.length !== 0) {
            htmlStream.write(space + "<div class='warnning' id='option_Crash'>\n");
            htmlStream.write(space + "  <h3>Warnning:</h3>\n");

            for (let i = 0; i < crashData.length; i++) {
                htmlStream.write(space + "  <p id='" + crashData[i] + "'>Crash happened when testing " +
                                 crashData[i] + ", please double check.</p>\n");
            }

            htmlStream.write(space + "  <hr />\n");
            htmlStream.write(space + "</div>\n");
        }
    }

    var createHtmlBodyContainerNewTestCase = function(space) {
        if (newTestCaseData.get("caseCount") !== 0) {
            htmlStream.write(space + "<hr />\n");

            htmlStream.write(space + "<div class='NewTestCase'>\n");
            htmlStream.write(space + "  <h3>NOTE: There are " + newTestCaseData.get("caseCount") +
                             " new test cases compared with the baseline, please double check.</h3>\n");
            htmlStream.write(space + "</div>\n");

            htmlStream.write(space + "<table>\n");
            htmlStream.write(space + "  <thead>\n");
            htmlStream.write(space + "    <tr>\n");
            htmlStream.write(space + "      <th>Feature\n");
            htmlStream.write(space + "      </th>\n");
            htmlStream.write(space + "      <th>TestCase\n");
            htmlStream.write(space + "      </th>\n");

            for (let backend of newTestCaseData.get("backends").keys()) {
                htmlStream.write(space + "      <th>" + backend + "\n");
                htmlStream.write(space + "      </th>\n");
            }

            htmlStream.write(space + "    </tr>\n");
            htmlStream.write(space + "  </thead>\n");
            htmlStream.write(space + "  <tbody>\n");

            for (let caseName of newTestCaseData.keys()) {
                if (caseName !== "caseCount" && caseName !== "backends") {
                    htmlStream.write(space + "    <tr >\n");
                    htmlStream.write(space + "      <td >" + newTestCaseData.get(caseName).get("title") + "\n");
                    htmlStream.write(space + "      </td>\n");
                    htmlStream.write(space + "      <td >" + newTestCaseData.get(caseName).get("caseID") + "\n");
                    htmlStream.write(space + "      </td>\n");

                    for (let backend of newTestCaseData.get("backends").keys()) {
                        if (newTestCaseData.get(caseName).get("backend").has(backend)) {
                            if (newTestCaseData.get(caseName).get("backend").get(backend) == "Pass") {
                                htmlStream.write(space + "      <td class='pass'>" +
                                                 newTestCaseData.get(caseName).get("backend").get(backend) + "\n");
                                htmlStream.write(space + "      </td>\n");
                            } else {
                                htmlStream.write(space + "      <td class='fail'>" +
                                                 newTestCaseData.get(caseName).get("backend").get(backend) + "\n");
                                htmlStream.write(space + "          </td>\n");
                            }
                        }
                    }

                    htmlStream.write(space + "    </tr>\n");
                }
            }

            htmlStream.write(space + "  </tbody>\n");
            htmlStream.write(space + "</table>\n");

            htmlStream.write(space + "<hr />\n");
        }
    }

    var createHtmlBodyContainerSuggest = function(space) {
        htmlStream.write(space + "<div>\n");
        htmlStream.write(space + "  <h3>PR Submission Proposal:</h3>\n");

        for (let testBackend of testBackends) {
            numberPasstoFail = numberPasstoFail + pageData.get(testBackend).get("pass2fail").length;
            numberFailtoPass = numberFailtoPass + pageData.get(testBackend).get("fail2pass").length;

            if (typeof pageDataTotal.get(testBackend).get("grasp")[0] !== "undefined") {
                numberTotal = numberTotal + pageDataTotal.get(testBackend).get("grasp")[0];
            }

            if (pageData.get(testBackend).get("pass2fail").length !== 0) {
                htmlStream.write(space + "    <h4>&emsp; &emsp; &#10148 &emsp; " + testBackend +
                                 ": <span class='notsuggest'>Please improve the code</span></h4>\n");
            } else {
                htmlStream.write(space + "    <h4>&emsp; &emsp; &#10148 &emsp; " + testBackend +
                                 ": <span class='suggest'>OK</span></h4>\n");
            }
        }

        htmlStream.write(space + "  <h3>PR Submission Message:</h3>\n");
        htmlStream.write(space + "    <div>Total Test Cases: " + numberTotal + "</div>\n");
        htmlStream.write(space + "    <div>Pass to Fail: " + numberPasstoFail + "</div>\n");
        htmlStream.write(space + "    <div>Fail to Pass: " + numberFailtoPass + "</div>\n");
        htmlStream.write(space + "  <hr />\n");
        htmlStream.write(space + "</div>\n");
    }

    var createHtmlBodyContainerResultMenu =  function(space) {
        htmlStream.write(space + "<div class='tab-menu'>\n");
        htmlStream.write(space + "  <ul>\n");
        htmlStream.write(space + "    <li class='active' id='tab_menu1' onclick='javascript:tab1_click()'>Pass2Fail</li>\n");
        htmlStream.write(space + "    <li id='tab_menu2' onclick='javascript:tab2_click()'>Fail2Pass</li>\n");
        htmlStream.write(space + "  </ul>\n");
        htmlStream.write(space + "</div>\n");
    }

    var createHtmlBodyContainerResultBoxTable =  function(space, backend) {
        htmlStream.write(space + "<table>\n");
        htmlStream.write(space + "  <thead>\n");
        htmlStream.write(space + "    <tr>\n");
        htmlStream.write(space + "      <th>Feature\n");
        htmlStream.write(space + "      </th>\n");
        htmlStream.write(space + "      <th>TestCase\n");
        htmlStream.write(space + "      </th>\n");
        htmlStream.write(space + "      <th>Baseline\n");
        htmlStream.write(space + "      </th>\n");
        htmlStream.write(space + "      <th>" + backend + "\n");
        htmlStream.write(space + "      </th>\n");
        htmlStream.write(space + "    </tr>\n");
        htmlStream.write(space + "  </thead>\n");
        htmlStream.write(space + "  <tbody>\n");

        if (pageData.get(backend).get("pass2fail").length == 0) {
            htmlStream.write(space + "    <tr class='pass2fail'>\n");
            htmlStream.write(space + "      <td colspan='4'>None changed\n");
            htmlStream.write(space + "      </td>\n");
            htmlStream.write(space + "    </tr>\n");
        } else {
            for (let i = 0; i < pageData.get(backend).get("pass2fail").length; i++) {
                htmlStream.write(space + "      <tr class='pass2fail'>\n");
                htmlStream.write(space + "        <td >" + pageData.get(backend).get("pass2fail")[i][0] + "\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "        <td >" + pageData.get(backend).get("pass2fail")[i][1] + "\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "        <td class='pass'>Pass\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "        <td class='fail'>Fail\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "      </tr>\n");
            }
        }

        if (pageData.get(backend).get("fail2pass").length == 0) {
            htmlStream.write(space + "    <tr class='fail2pass'>\n");
            htmlStream.write(space + "      <td colspan='4'>None changed\n");
            htmlStream.write(space + "      </td>\n");
            htmlStream.write(space + "    </tr>\n");
        } else {
            for (let i = 0; i < pageData.get(backend).get("fail2pass").length; i++) {
                htmlStream.write(space + "      <tr class='fail2pass'>\n");
                htmlStream.write(space + "        <td >" + pageData.get(backend).get("fail2pass")[i][0] + "\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "        <td >" + pageData.get(backend).get("fail2pass")[i][1] + "\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "        <td class='fail'>Fail\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "        <td class='pass'>Pass\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "      </tr>\n");
            }
        }

        htmlStream.write(space + "  </tbody>\n");
        htmlStream.write(space + "</table><br /><br />\n");
    }

    var createHtmlBodyContainerResultBoxTableTotal =  function(space) {
        htmlStream.write(space + "<table>\n");
        htmlStream.write(space + "  <thead>\n");
        htmlStream.write(space + "    <tr>\n");
        htmlStream.write(space + "      <th rowspan='2'>Summary\n");
        htmlStream.write(space + "      </th>\n");
        for (let i = 0; i < testBackends.length; i++) {
            htmlStream.write(space + "      <th colspan='2'>" + testBackends[i] + "\n");
            htmlStream.write(space + "      </th>\n");
        }

        htmlStream.write(space + "    </tr>\n");
        htmlStream.write(space + "    <tr>\n");
        for (let i = 0; i < testBackends.length; i++) {
            htmlStream.write(space + "      <th>Baseline\n");
            htmlStream.write(space + "      </th>\n");
            htmlStream.write(space + "      <th>Test Build\n");
            htmlStream.write(space + "      </th>\n");
        }

        htmlStream.write(space + "    </tr>\n");
        htmlStream.write(space + "  </thead>\n");
        htmlStream.write(space + "  <tbody>\n");

        let TableTotalDataArray = ["Total", "Pass", "Fail", "Block", "PassRate%"];
        for (let i = 0; i < TableTotalDataArray.length; i++) {
            htmlStream.write(space + "    <tr>\n");
            htmlStream.write(space + "      <th>" + TableTotalDataArray[i] + "\n");
            htmlStream.write(space + "      </th>\n");

            for (let j = 0; j < testBackends.length; j++) {
                htmlStream.write(space + "      <td>" + pageDataTotal.get(testBackends[j]).get("Baseline")[i] + "\n");
                htmlStream.write(space + "      </td>\n");

                if (typeof pageDataTotal.get(testBackends[j]).get("grasp")[i] == "undefined") {
                    htmlStream.write(space + "      <td>N/A\n");
                } else {
                    htmlStream.write(space + "      <td>" + pageDataTotal.get(testBackends[j]).get("grasp")[i] + "\n");
                }

                htmlStream.write(space + "      </td>\n");
            }

            htmlStream.write(space + "    </tr>\n");
        }

        htmlStream.write(space + "  </tbody>\n");
        htmlStream.write(space + "</table>\n");
    }

    var createHtmlBodyContainerResultBox =  function(space) {
        htmlStream.write(space + "<div class='tab-box'>\n");
        htmlStream.write(space + "  <div class='active' id='tab_box'>\n");

        for (let i = 0; i < testBackends.length; i++) {
            let flag = false;

            for (let j = 0; j < crashData.length; j++) {
                if (testBackends[i] == crashData[j]) flag = true;
            }

            if (crashData.length !== 0 && flag) {
                continue;
            } else {
                createHtmlBodyContainerResultBoxTable(space + "    ", testBackends[i]);
            }
        }

        createHtmlBodyContainerResultBoxTableTotal(space + "    ");
        createHtmlBodyContainerNewTestCase(space + "    ");

        htmlStream.write(space + "  </div>\n");
        htmlStream.write(space + "</div>\n");
    }

    var createHtmlBodyContainerResult = function(space) {
        htmlStream.write(space + "<h3>Result:</h3>\n");

        createHtmlBodyContainerResultMenu(space);
        createHtmlBodyContainerResultBox(space);
    }

    var createHtmlBodyContainer = function(space) {
        htmlStream.write(space + "<div class='container'>\n");

        createHtmlBodyContainerVersion(space + "  ");
        createHtmlBodyContainerCrash(space + "  ");
        createHtmlBodyContainerSuggest(space + "  ");
        createHtmlBodyContainerResult(space + "  ");

        htmlStream.write(space + "</div>\n");
    }

    var createHtmlBody = function(space) {
        htmlStream.write(space + "<body>\n");

        createHtmlBodyContainer(space + "  ");

        htmlStream.write(space + "</body>\n");
    }

    var createHtmlFile = function() {
        fs.writeFileSync(htmlPath, "<!DOCTYPE html>\n");

        htmlStream.write("<html>\n");

        createHtmlHead();
        createHtmlBody("  ");

        htmlStream.write("</html>\n");
    }

    RClog("time", "mark");

    for (let i = 0; i < backendModels.length; i++) {
        chromeOption = new Chrome.Options();
        backendModel = backendModels[i];
        graspDataSummary["total"] = 0;
        graspDataSummary["pass"] = 0;
        graspDataSummary["fail"] = 0;
        graspDataSummary["block"] = 0;
        continueFlag = false;
        remoteURL = "https://brucedai.github.io/nt/test/index-local.html";

        if (backendModel === "Mac-MPS") {
            if (testPlatform === "Mac") {
                testBackends.push("Mac-MPS");
                remoteURL = remoteURL + "?backend=mps";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (backendModel === "Mac-BNNS") {
            if (testPlatform === "Mac") {
                testBackends.push("Mac-BNNS");
                remoteURL = remoteURL + "?backend=bnns";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (backendModel === "Mac-WASM") {
            if (testPlatform === "Mac") {
                testBackends.push("Mac-WASM");
                remoteURL = remoteURL + "?backend=wasm";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (backendModel === "Mac-WebGL") {
            if (testPlatform === "Mac") {
                testBackends.push("Mac-WebGL");
                remoteURL = remoteURL + "?backend=webgl";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (backendModel === "Android-NNAPI") {
            if (testPlatform === "Android") {
                testBackends.push("Android-NNAPI");
                remoteURL = remoteURL + "?backend=nnapi";
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--enable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (backendModel === "Android-WASM") {
            if (testPlatform === "Android") {
                testBackends.push("Android-WASM");
                remoteURL = remoteURL + "?backend=wasm";
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--disable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (backendModel === "Android-WebGL") {
            if (testPlatform === "Android") {
                testBackends.push("Android-WebGL");
                remoteURL = remoteURL + "?backend=webgl";
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--disable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (backendModel === "Windows-clDNN") {
            if (testPlatform === "Windows") {
                testBackends.push("Windows-clDNN");
                remoteURL = remoteURL + "?backend=cldnn";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (backendModel === "Windows-WASM") {
            if (testPlatform === "Windows") {
                testBackends.push("Windows-WASM");
                remoteURL = remoteURL + "?backend=wasm";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (backendModel === "Windows-WebGL") {
            if (testPlatform === "Windows") {
                testBackends.push("Windows-WebGL");
                remoteURL = remoteURL + "?backend=webgl";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (backendModel === "Linux-clDNN") {
            if (testPlatform === "Linux") {
                testBackends.push("Linux-clDNN");
                remoteURL = remoteURL + "?backend=cldnn";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (backendModel === "Linux-WASM") {
            if (testPlatform === "Linux") {
                testBackends.push("Linux-WASM");
                remoteURL = remoteURL + "?backend=wasm";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (backendModel === "Linux-WebGL") {
            if (testPlatform === "Linux") {
                testBackends.push("Linux-WebGL");
                remoteURL = remoteURL + "?backend=webgl";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        }

        driver = new Builder()
            .forBrowser("chrome")
            .setChromeOptions(chromeOption)
            .build();

        await driver.get(remoteURL);
        await driver.wait(until.elementLocated(By.xpath("//*[@id='mocha-stats']/li[1]/canvas")), 100000).then(function() {
            RClog("console", "open remote URL: " + remoteURL);
        }).catch(function() {
            throw new Error("failed to load web page");
        });

        RClog("time", "mark");

        await driver.wait(async function() {
            await driver.executeScript("return window.mochaFinish;").then(function(flag) {
                RClog("debug", flag);
            }).catch(function(err) {
                throw err;
            });

            return driver.executeScript("return window.mochaFinish;").catch(function(err) {
                throw err;
            });
        }, 200000).then(function() {
            RClog("console", "load remote URL is completed, no crash");
        }).catch(function(err) {
            RClog("debug", err);

            if (err.message.search("session deleted because of page crash") != -1) {
                continueFlag = true;
                crashData.push(backendModel);
                RClog("console", "remote URL is crashed");
            } else {
                throw err;
            }
        });

        RClog("time", "mark");

        if (continueFlag) {
            await driver.sleep(2000);
            await driver.quit();
            await driver.sleep(2000);

            continue;
        }

        RClog("console", "checking with '" + backendModel + "' backend is start");

        RClog("console", "checking....");

        await graspResult();

        RClog("time", "mark");

        pageDataTotal.get(backendModel).get("grasp").push(graspDataSummary["total"]);
        pageDataTotal.get(backendModel).get("grasp").push(graspDataSummary["pass"]);
        pageDataTotal.get(backendModel).get("grasp").push(graspDataSummary["fail"]);
        pageDataTotal.get(backendModel).get("grasp").push(graspDataSummary["block"]);
        pageDataTotal.get(backendModel).get("grasp").push(Math.round((graspDataSummary["pass"] / graspDataSummary["total"]) * 100).toString() + "%");

        await driver.sleep(2000);
        await driver.quit();
        await driver.sleep(2000);

        RClog("console", "checking with '" + backendModel + "' backend is completed");
    }

    await createHtmlFile();

    htmlStream.end();

    if (testPlatform == "Android") {
        driver = new Builder()
            .forBrowser("chrome")
            .setChromeOptions(new Chrome.Options().androidPackage("org.chromium.chrome").androidDeviceSerial(androidSN))
            .build();

        await driver.sleep(3000);
        driver.quit();
        await driver.sleep(3000);

        command = adbPath + " kill-server";
        execSync(command, {encoding: "UTF-8", stdio: "pipe"});
    }

    driver = new Builder()
        .forBrowser("chrome")
        .setChromeOptions(new Chrome.Options().setChromeBinaryPath(chromiumPath))
        .build();

    if (sys == "Windows_NT") {
        htmlPath = "file://" + process.cwd() + "\\output\\report-check-result.html";
    } else {
        htmlPath = "file://" + process.cwd() + "/output/report-check-result.html";
    }

    RClog("time", "mark");

    await driver.get(htmlPath);
})().then(function() {
    RClog("console", "checking chromium code is completed");
}).catch(function(err) {
    driver.quit();
    RClog("console", err);
});
