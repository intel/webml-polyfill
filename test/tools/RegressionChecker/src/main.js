const Builder = require("../node_modules/selenium-webdriver").Builder;
const By = require("../node_modules/selenium-webdriver").By;
const until = require("../node_modules/selenium-webdriver").until;
const Chrome = require("../node_modules/selenium-webdriver/chrome");
const csv = require("../node_modules/fast-csv");
const execSync = require("child_process").execSync;
const fs = require("fs");
const os = require("os");
require("chromedriver");

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

var outputPath, debugPath, resultHTMLPath;
if (os.type() == "Windows_NT") {
    outputPath = ".\\output";
    debugPath = ".\\output\\debug";
    resultHTMLPath = outputPath + "\\report-check-result.html";
    resultHTMLPathFull = "file://" + process.cwd() + "\\output\\report-check-result.html";
} else {
    outputPath = "./output";
    debugPath = "./output/debug";
    resultHTMLPath = outputPath + "/report-check-result.html";
    resultHTMLPathFull = "file://" + process.cwd() + "/output/report-check-result.html";
}

if (!fs.existsSync(outputPath)) {
    fs.mkdirSync(outputPath);
}

if (!fs.existsSync(debugPath)) {
    fs.mkdirSync(debugPath);
}

var resultHTMLStream = fs.createWriteStream(resultHTMLPath, {flags: "a"});

RClog("time", "mark");
RClog("console", "checking config.json file....");

function jsonTypeCheck (json, field, expectType) {
    if (typeof json[field] == expectType) {
        return true;
    } else {
        throw new Error("Type of 'JSON." + field + "' is not '" + expectType + "'");
    }
}

var testPlatform, chromiumPath, preferIEMYRIAD, supportSwitch, webmlPolyfill, webnn;
var RCjson = JSON.parse(fs.readFileSync("./config.json"));
if (jsonTypeCheck(RCjson, "platform", "string")) {
    testPlatform = RCjson.platform;
}

if (jsonTypeCheck(RCjson, "chromiumPath", "string")) {
    chromiumPath = RCjson.chromiumPath;
}

if (jsonTypeCheck(RCjson, "IEMYRIAD", "boolean")) {
    preferIEMYRIAD = RCjson.IEMYRIAD;
}

if (jsonTypeCheck(RCjson, "supportSwitch", "boolean")) {
    supportSwitch = RCjson.supportSwitch;
}

if (jsonTypeCheck(RCjson, "webmlPolyfill", "boolean")) {
    webmlPolyfill = RCjson.webmlPolyfill;
}

if (jsonTypeCheck(RCjson, "webnn", "boolean")) {
    webnn = RCjson.webnn;
}

var testPrefers = new Array();
if (testPlatform == "Linux") {
    if (preferIEMYRIAD) {
        testPrefers.push("Linux-WebNN-Low-IE-MYRIAD");
    } else {
        if (webmlPolyfill) {
            testPrefers.push("Linux-Polyfill-Fast-WASM");
            testPrefers.push("Linux-Polyfill-Sustained-WebGL");
        }

        if (webnn) {
            testPrefers.push("Linux-WebNN-Fast-MKLDNN");
            testPrefers.push("Linux-WebNN-Sustained-clDNN");

            if (supportSwitch) {
                testPrefers.push("Linux-WebNN-Fast-IE-MKLDNN");
                testPrefers.push("Linux-WebNN-Sustained-IE-clDNN");
            }
        }
    }
} else if (testPlatform == "Android") {
    if (webmlPolyfill) {
        testPrefers.push("Android-Polyfill-Fast-WASM");
        testPrefers.push("Android-Polyfill-Sustained-WebGL");
    }

    if (webnn) {
        testPrefers.push("Android-WebNN-Sustained-NNAPI");
    }
} else if (testPlatform == "Mac") {
    if (webmlPolyfill) {
        testPrefers.push("macOS-Polyfill-Fast-WASM");
        testPrefers.push("macOS-Polyfill-Sustained-WebGL");
    }

    if (webnn) {
        testPrefers.push("macOS-WebNN-Fast-BNNS");
        testPrefers.push("macOS-WebNN-Sustained-MPS");

        if (supportSwitch) {
            testPrefers.push("macOS-WebNN-Fast-MKLDNN");
        }
    }
} else if (testPlatform == "Windows") {
    if (webmlPolyfill) {
        testPrefers.push("Win-Polyfill-Fast-WASM");
        testPrefers.push("Win-Polyfill-Sustained-WebGL");
    }

    if (webnn) {
        testPrefers.push("Win-WebNN-Fast-MKLDNN");
        testPrefers.push("Win-WebNN-Sustained-clDNN");

        if (supportSwitch) {
            testPrefers.push("Win-WebNN-Sustained-DML");
            testPrefers.push("Win-WebNN-Low-DML");
        }
    }
}

RClog("console", "prefers: " + testPrefers);

RClog("time", "mark");
RClog("console", "checking baseline files....");

var baselinejson = JSON.parse(fs.readFileSync("./baseline/baseline.config.json"));

for (let prefer of testPrefers) {
    if (!Object.keys(baselinejson).includes(prefer)) {
        let str = "No baseline data in baseline.config.json file: " + prefer;
        throw new Error(str);
    }
}

/**
 * pageData = {
 *     prefer: {
 *         "pass2fail": [value1, value2, value3],
 *         "fail2pass": [value1, value2, value3]
 *     }
 * }
 */
var pageData = new Map();
/**
 * pageDataTotal = {
 *     prefer: {
 *         "Baseline": [total, pass, fail, block, passrate],
 *         "grasp": [total, pass, fail, block, passrate]
 *     }
 * }
 */
var pageDataTotal = new Map();
var versionChromium, versionPolyfill;

for (let [key1, value1] of Object.entries(baselinejson)) {
    if (key1 == "Version") {
        for (let key2 of Object.keys(value1)) {
            if (key2 == "chromium") versionChromium = baselinejson[key1][key2];
            if (key2 == "polyfill") versionPolyfill = baselinejson[key1][key2];
        }
    } else {
        if (testPrefers.includes(key1)) {
            pageData.set(key1, new Map([["pass2fail", new Array()], ["fail2pass", new Array()]]));
            pageDataTotal.set(key1, new Map([["Baseline", new Array(
                baselinejson[key1]["total"],
                baselinejson[key1]["pass"],
                baselinejson[key1]["fail"],
                baselinejson[key1]["block"],
                Math.round((baselinejson[key1]["pass"] / baselinejson[key1]["total"]) * 100).toString() + "%"
            )], ["grasp", new Array()]]));
        }
    }
}

/**
 * baseLineData = {
 *     key: {
 *         "Feature": value,
 *         "CaseId": value,
 *         "TestCase": value,
 *         "macOS-Polyfill-Fast-WASM": value,
 *         "macOS-Polyfill-Sustained-WebGL": value,
 *         "macOS-WebNN-Fast-BNNS": value,
 *         "macOS-WebNN-Fast-MKLDNN": value,
 *         "macOS-WebNN-Sustained-MPS": value,
 *         "Android-Polyfill-Fast-WASM": value,
 *         "Android-Polyfill-Sustained-WebGL": value,
 *         "Android-WebNN-Fast-NNAPI": value,
 *         "Android-WebNN-Sustained-NNAPI": value,
 *         "Android-WebNN-Low-NNAPI": value,
 *         "Win-Polyfill-Fast-WASM": value,
 *         "Win-Polyfill-Sustained-WebGL": value,
 *         "Win-WebNN-Fast-MKLDNN": value,
 *         "Win-WebNN-Sustained-DML": value,
 *         "Win-WebNN-Sustained-clDNN": value,
 *         "Win-WebNN-Low-DML": value,
 *         "Linux-Polyfill-Fast-WASM": value,
 *         "Linux-Polyfill-Sustained-WebGL": value,
 *         "Linux-WebNN-Fast-MKLDNN": value,
 *         "Linux-WebNN-Sustained-clDNN": value,
 *         "Linux-WebNN-Fast-IE-MKLDNN": value,
 *         "Linux-WebNN-Sustained-IE-clDNN": value,
 *         "Linux-WebNN-Low-IE-MYRIAD": value
 *     }
 * }
 */
var baseLineData = new Map();

/**
 * baseLineDataSummary = [
 *     model-name1: count1,
 *     model-name2: count2,
 *     model-name3: count3
 * ]
 */
var baseLineDataSummary = new Map();

csv.fromPath("./baseline/unitTestsBaseline.csv", {headers: true})
.on("data", function(data) {
    let keyArray = new Array();
    let valueArray = new Array();

    for (let [key, value] of Object.entries(data)) {
        keyArray.push(key);
        valueArray.push(value);
    }

    for (let prefer of testPrefers) {
        if (!keyArray.includes(prefer)) {
            let str = "No baseline data in unitTestsBaseline.csv file: " + prefer;
            throw new Error(str);
        }
    }

    let newData = valueArray[1].split("/")[0];
    let dataArray = valueArray[1].split("/");

    if (dataArray.length > 2) {
        for (let dataCount = 1; dataCount < dataArray.length - 1; dataCount++) {
            newData = newData + "/" + dataArray[dataCount];
        }
    }

    baseLineData.set(valueArray[0] + "-" + valueArray[1], new Map());
    baseLineData.set(valueArray[0] + "-" + newData + "-" + valueArray[2], new Map());

    for (let x in keyArray) {
        if (keyArray[x] == "Case Id") {
            baseLineData.get(valueArray[0] + "-" + valueArray[1]).set("CaseId", valueArray[x]);
            baseLineData.get(valueArray[0] + "-" + newData + "-" + valueArray[2]).set("CaseId", newData);
        } else if (keyArray[x] == "Test Case") {
            baseLineData.get(valueArray[0] + "-" + valueArray[1]).set("TestCase", valueArray[x]);
            baseLineData.get(valueArray[0] + "-" + newData + "-" + valueArray[2]).set("TestCase", valueArray[x]);
        } else {
            baseLineData.get(valueArray[0] + "-" + valueArray[1]).set(keyArray[x], valueArray[x]);
            baseLineData.get(valueArray[0] + "-" + newData + "-" + valueArray[2]).set(keyArray[x], valueArray[x]);
        }
    }

    if (baseLineDataSummary.has(valueArray[0] + "-" + newData)) {
        baseLineDataSummary.set(valueArray[0] + "-" + newData, baseLineDataSummary.get(valueArray[0] + "-" + newData) + 1);
    } else {
        baseLineDataSummary.set(valueArray[0] + "-" + newData, 1);
    }
})
.on("end", function() {
    for (let key of baseLineData.keys()) {
        RClog("debug", "key: " + key);
    }

    for (let key of baseLineDataSummary.keys()) {
        RClog("debug", "key: " + key);
        RClog("debug", "value: " + baseLineDataSummary.get(key));
    }
});

var crashData = new Array();
/**
 * newTestCaseData = {
 *     "caseCount": value,
 *     "prefers": {
 *         prefer1: true,
 *         prefer2: true
 *     },
 *     testCase: {
 *         "title": title,
 *         "caseID": caseID,
 *         "prefer": {
 *             prefer1: caseStatus1,
 *             prefer2: caseStatus2,
 *             prefer3: caseStatus3
 *         }
 *     }
 * }
 */
var newTestCaseData = new Map();
newTestCaseData.set("caseCount", 0);
newTestCaseData.set("prefers", new Map());

/**
 * graspDataSummary = [
 *     total: value,
 *     pass: value,
 *     fail: value,
 *     block: value
 * ]
 */
var graspDataSummary = new Array();

RClog("console", "checking runtime environment....");
RClog("time", "mark");

var command, androidSN, adbPath;
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

var remoteURL, driver, chromeOption, testPrefer;
var continueFlag = false;
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

            let baseLineStatus = baseLineData.get(key).get(testPrefer);
            if (caseStatus !== baseLineStatus) {
                if (baseLineStatus == "Pass" && caseStatus == "Fail") {
                    pageData.get(testPrefer).get("pass2fail").push([title, module + "-" + caseName]);
                } else {
                    pageData.get(testPrefer).get("fail2pass").push([title, module + "-" + caseName]);
                }

                RClog("debug", title + "-" + module + "-" + caseName);
                RClog("debug", "baseLineStatus: " + baseLineStatus + " - caseStatus: " + caseStatus);
            }
        } else {
            RClog("debug", "no match test case: " + title + "-" + module + "-" + caseName);

            if (matchFlag == "macth" || matchFlag == "delete") {
                throw new Error("no match test case: " + title + "-" + module + "-" + caseName);
            } else {
                if (!newTestCaseData.get("prefers").has(testPrefer)) {
                    newTestCaseData.get("prefers").set(testPrefer, true);
                }

                if (!newTestCaseData.has(title + "-" + module + "-" + caseName)) {
                    newTestCaseData.set(title + "-" + module + "-" + caseName, new Map());
                    newTestCaseData.get(title + "-" + module + "-" + caseName).set("title", title);
                    newTestCaseData.get(title + "-" + module + "-" + caseName).set("caseID", module + "-" + caseName);
                    newTestCaseData.get(title + "-" + module + "-" + caseName).set("prefer", new Map());
                    newTestCaseData.set("caseCount", newTestCaseData.get("caseCount") + 1);
                }

                newTestCaseData.get(title + "-" + module + "-" + caseName).get("prefer").set(testPrefer, caseStatus);
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
        if (graspTotal == baselinejson[testPrefer]["total"]) {
            matchFlag = "macth";
            RClog("console", "match base line data: matching mode");
        } else {
            if (graspTotal > baselinejson[testPrefer]["total"]) {
                matchFlag = "add";
            } else if (graspTotal < baselinejson[testPrefer]["total"]) {
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
    .box-menu {margin: 10px 0px -10px 0px;}\n\
    .box-menu ul {height:30px;border-bottom:1px solid gray;list-style:none;padding-left:0;}\n\
    .box-menu ul li {float:left;width:150px;margin-right:3px;color:#000;border:solid 1px gray;border-bottom:none; text-align:center;line-height:30px;}\n\
    .box-menu ul li.active {background-color: #007bc7;color: #fff;}\n\
    .box-menu ul li:hover {cursor: pointer;}\n\
    .box-menu ul ul {height:30px;border-bottom:1px solid gray;list-style:none;padding-left:0;}\n\
    .box-menu ul ul li {float:right;width:150px;margin-right:3px;color:#000;border:solid 1px gray;border-bottom:none; text-align:center;line-height:30px;}\n\
    .box-menu ul ul li.active {background-color: #007bc7;color: #fff;}\n\
    .box-menu ul ul li:hover {cursor: pointer;}\n\
    .box-table div {display:none;}\n\
    .box-table div.active {display:block;}\n\
    .box-table div.box-table-all {display:block;}\n\
    .box-table div.box-table-new {display:block;}\n\
    table {border: 1px solid #ddd; border-spacing:0;}\n\
    table tr th {border: 1px solid #000;background-color: #B0C4DE;}\n\
    table tr td {border: 1px solid #ddd}\n\
    table tr:nth-child(even){background: #F0F0F0;}\n\
    table tr td.box-table-log-number {width:20px;overflow:hidden;text-align: center;text-overflow: ellipsis;border: 2px solid #FFFFFF;\
        border-top: none;border-bottom: none;border-left: none;}\n\
    table tr td.box-table-log-text {overflow:hidden;border:none;}\n\
    table.box-table-log {border:1px solid #ddd;}\n\
    .warnning {color:red}\n\
    .pass {color:green}\n\
    .fail {color:red}\n\
    </style>\n\
    <script>\n\
      function click_box_menu(data) {\n\
        var keyWord = data.getAttribute('data-info');\n\
        var keyWordsAll = new Array();\n\
        keyWordsAll.push('pass2fail');\n\
        keyWordsAll.push('fail2pass');\n";

        resultHTMLStream.write(htmlDataHead);

        let prefers;
        for (let i = 0; i < testPrefers.length; i++) {
            if (i == 0) {
                prefers = "'" + testPrefers[i] + "'";
            } else {
                prefers = prefers + ",'" + testPrefers[i] + "'";
            }
        }

        resultHTMLStream.write("        let prefers = [" + prefers + "];\n");

        htmlDataHead = "\
        for (let prefer of prefers) {\n\
            keyWordsAll.push(prefer);\n\
        }\n\
        for (let key of keyWordsAll) {\n\
            let boxMenuKey = 'box-menu-' + key;\n\
            let boxTableKey = 'box-table-' + key;\n\
            if (key == keyWord) {\n\
                document.getElementById(boxMenuKey).classList.add('active');\n\
                document.getElementById(boxTableKey).classList.add('active');\n\
            } else {\n\
                document.getElementById(boxMenuKey).classList.remove('active');\n\
                document.getElementById(boxTableKey).classList.remove('active');\n\
            }\n\
        }\n\
      }\n\
    </script>\n\
  </head>\n";

        resultHTMLStream.write(htmlDataHead);
    }

    var bodyContainerVersion = function(space) {
        resultHTMLStream.write(space + "<div>\n");
        resultHTMLStream.write(space + "  <h2>PR Submission Checking Summary</h2>\n");
        resultHTMLStream.write(space + "  <hr />\n");
        resultHTMLStream.write(space + "  <h3>Baseline Information:</h3>\n");
        resultHTMLStream.write(space + "    <div>Chromium version: " + versionChromium + "</div>\n");
        resultHTMLStream.write(space + "    <div>Webml-polyfill version: " + versionPolyfill + "</div>\n");
        resultHTMLStream.write(space + "</div>\n");

        resultHTMLStream.write(space + "<hr />\n");
    }

    var bodyContainerCrash = function(space) {
        if (crashData.length !== 0) {
            resultHTMLStream.write(space + "<div class='warnning' id='option_Crash'>\n");
            resultHTMLStream.write(space + "  <h3>Warnning:</h3>\n");

            for (let i = 0; i < crashData.length; i++) {
                resultHTMLStream.write(space + "  <p id='" + crashData[i] + "'>Crash happened when testing " +
                                 crashData[i] + ", please double check.</p>\n");
            }

            resultHTMLStream.write(space + "  <hr />\n");
            resultHTMLStream.write(space + "</div>\n");
        }
    }

    var bodyContainerNewTest = function(space) {
        if (newTestCaseData.get("caseCount") !== 0) {
            resultHTMLStream.write(space + "<hr />\n");

            resultHTMLStream.write(space + "<div class='box-table-new'>\n");
            resultHTMLStream.write(space + "  <h3>NOTE: There are " + newTestCaseData.get("caseCount") +
                             " new test cases compared with the baseline, please double check.</h3>\n");

            resultHTMLStream.write(space + "  <table>\n");
            resultHTMLStream.write(space + "    <thead>\n");
            resultHTMLStream.write(space + "      <tr>\n");
            resultHTMLStream.write(space + "        <th>Feature\n");
            resultHTMLStream.write(space + "        </th>\n");
            resultHTMLStream.write(space + "        <th>TestCase\n");
            resultHTMLStream.write(space + "        </th>\n");

            for (let prefer of newTestCaseData.get("prefers").keys()) {
                resultHTMLStream.write(space + "        <th>" + prefer + "\n");
                resultHTMLStream.write(space + "        </th>\n");
            }

            resultHTMLStream.write(space + "      </tr>\n");
            resultHTMLStream.write(space + "    </thead>\n");
            resultHTMLStream.write(space + "    <tbody>\n");

            for (let caseName of newTestCaseData.keys()) {
                if (caseName !== "caseCount" && caseName !== "prefers") {
                    resultHTMLStream.write(space + "      <tr >\n");
                    resultHTMLStream.write(space + "        <td >" + newTestCaseData.get(caseName).get("title") + "\n");
                    resultHTMLStream.write(space + "        </td>\n");
                    resultHTMLStream.write(space + "        <td >" + newTestCaseData.get(caseName).get("caseID") + "\n");
                    resultHTMLStream.write(space + "        </td>\n");

                    for (let prefer of newTestCaseData.get("prefers").keys()) {
                        if (newTestCaseData.get(caseName).get("prefer").has(prefer)) {
                            if (newTestCaseData.get(caseName).get("prefer").get(prefer) == "Pass") {
                                resultHTMLStream.write(space + "        <td class='pass'>" +
                                                 newTestCaseData.get(caseName).get("prefer").get(prefer) + "\n");
                                resultHTMLStream.write(space + "        </td>\n");
                            } else {
                                resultHTMLStream.write(space + "        <td class='fail'>" +
                                                 newTestCaseData.get(caseName).get("prefer").get(prefer) + "\n");
                                resultHTMLStream.write(space + "            </td>\n");
                            }
                        }
                    }

                    resultHTMLStream.write(space + "      </tr>\n");
                }
            }

            resultHTMLStream.write(space + "    </tbody>\n");
            resultHTMLStream.write(space + "  </table>\n");
            resultHTMLStream.write(space + "</div>\n");

            resultHTMLStream.write(space + "<hr />\n");
        }
    }

    var bodyContainerSuggest = function(space) {
        resultHTMLStream.write(space + "<div>\n");
        resultHTMLStream.write(space + "  <h3>PR Submission Proposal:</h3>\n");

        for (let testPrefer of testPrefers) {
            numberPasstoFail = numberPasstoFail + pageData.get(testPrefer).get("pass2fail").length;
            numberFailtoPass = numberFailtoPass + pageData.get(testPrefer).get("fail2pass").length;

            if (typeof pageDataTotal.get(testPrefer).get("grasp")[0] !== "undefined") {
                numberTotal = numberTotal + pageDataTotal.get(testPrefer).get("grasp")[0];
            }

            if (pageData.get(testPrefer).get("pass2fail").length !== 0) {
                resultHTMLStream.write(space + "    <h4>&emsp; &emsp; &#10148 &emsp; " + testPrefer +
                                 ": <span class='notsuggest'>Please improve the code</span></h4>\n");
            } else {
                resultHTMLStream.write(space + "    <h4>&emsp; &emsp; &#10148 &emsp; " + testPrefer +
                                 ": <span class='suggest'>OK</span></h4>\n");
            }
        }

        resultHTMLStream.write(space + "  <h3>PR Submission Message:</h3>\n");
        resultHTMLStream.write(space + "    <div>Total Test Cases: " + numberTotal + "</div>\n");
        resultHTMLStream.write(space + "    <div>Pass to Fail: " + numberPasstoFail + "</div>\n");
        resultHTMLStream.write(space + "    <div>Fail to Pass: " + numberFailtoPass + "</div>\n");
        resultHTMLStream.write(space + "  <hr />\n");
        resultHTMLStream.write(space + "</div>\n");
    }

    var bodyContainerBoxMenu =  function(space) {
        resultHTMLStream.write(space + "<div class='box-menu'>\n");
        resultHTMLStream.write(space + "  <ul>\n");
        resultHTMLStream.write(space + "    <li class='active' id='box-menu-pass2fail' data-info='pass2fail' onclick='javascript:click_box_menu(this)'>Pass2Fail</li>\n");
        resultHTMLStream.write(space + "    <li id='box-menu-fail2pass' data-info='fail2pass' onclick='javascript:click_box_menu(this)'>Fail2Pass</li>\n");
        resultHTMLStream.write(space + "    <ul>\n");

        for (let prefer of testPrefers) {
            if (prefer == "macOS-Polyfill-Fast-WASM" ||
            prefer == "macOS-Polyfill-Sustained-WebGL" ||
            prefer == "macOS-WebNN-Fast-BNNS" ||
            prefer == "macOS-WebNN-Fast-MKLDNN" ||
            prefer == "macOS-WebNN-Sustained-MPS" ||
            prefer == "Android-Polyfill-Fast-WASM" ||
            prefer == "Android-Polyfill-Sustained-WebGL" ||
            prefer == "Android-WebNN-Sustained-NNAPI" ||
            prefer == "Win-Polyfill-Fast-WASM" ||
            prefer == "Win-Polyfill-Sustained-WebGL" ||
            prefer == "Win-WebNN-Fast-MKLDNN" ||
            prefer == "Win-WebNN-Sustained-clDNN" ||
            prefer == "Linux-Polyfill-Fast-WASM" ||
            prefer == "Linux-Polyfill-Sustained-WebGL" ||
            prefer == "Linux-WebNN-Fast-MKLDNN" ||
            prefer == "Linux-WebNN-Sustained-clDNN") {
                resultHTMLStream.write(space + "      <li id='box-menu-" + prefer + "' data-info='" + prefer +
                "' onclick='javascript:click_box_menu(this)'>log-" + prefer.split("-")[3] + "</li>\n");
            } else if (prefer == "Win-WebNN-Sustained-DML" ||
            prefer == "Win-WebNN-Low-DML") {
                resultHTMLStream.write(space + "      <li id='box-menu-" + prefer + "' data-info='" + prefer +
                "' onclick='javascript:click_box_menu(this)'>log-" + prefer.split("-")[2] + "-" + prefer.split("-")[3] + "</li>\n");
            } else if (prefer == "Linux-WebNN-Fast-IE-MKLDNN" ||
            prefer == "Linux-WebNN-Sustained-IE-clDNN" ||
            prefer == "Linux-WebNN-Low-IE-MYRIAD") {
                resultHTMLStream.write(space + "      <li id='box-menu-" + prefer + "' data-info='" + prefer +
                "' onclick='javascript:click_box_menu(this)'>log-" + prefer.split("-")[3] + "-" + prefer.split("-")[4] + "</li>\n");
            }
        }

        resultHTMLStream.write(space + "    </ul>\n");
        resultHTMLStream.write(space + "  </ul>\n");
        resultHTMLStream.write(space + "</div>\n");
    }

    var bodyContainerBoxTablePrefer =  function(space, prefer, key) {
        resultHTMLStream.write(space + "<table>\n");
        resultHTMLStream.write(space + "  <thead>\n");
        resultHTMLStream.write(space + "    <tr>\n");
        resultHTMLStream.write(space + "      <th>Feature\n");
        resultHTMLStream.write(space + "      </th>\n");
        resultHTMLStream.write(space + "      <th>TestCase\n");
        resultHTMLStream.write(space + "      </th>\n");
        resultHTMLStream.write(space + "      <th>Baseline\n");
        resultHTMLStream.write(space + "      </th>\n");
        resultHTMLStream.write(space + "      <th>" + prefer + "\n");
        resultHTMLStream.write(space + "      </th>\n");
        resultHTMLStream.write(space + "    </tr>\n");
        resultHTMLStream.write(space + "  </thead>\n");
        resultHTMLStream.write(space + "  <tbody>\n");

        let keyArray = new Array();
        for (let baseLinekey of baseLineData.keys()) {
            for (let i = 0; i < pageData.get(prefer).get(key).length; i++) {
                if (baseLinekey == (pageData.get(prefer).get(key)[i][0] + "-" + pageData.get(prefer).get(key)[i][1])) {
                    keyArray.push([pageData.get(prefer).get(key)[i][0], pageData.get(prefer).get(key)[i][1]]);
                }
            }
        }

        if (pageData.get(prefer).get(key).length == 0) {
            resultHTMLStream.write(space + "    <tr>\n");
            resultHTMLStream.write(space + "      <td colspan='4'>None changed\n");
            resultHTMLStream.write(space + "      </td>\n");
            resultHTMLStream.write(space + "    </tr>\n");
        } else {
            for (let i = 0; i < keyArray.length; i++) {
                resultHTMLStream.write(space + "      <tr>\n");
                resultHTMLStream.write(space + "        <td >" + keyArray[i][0] + "\n");
                resultHTMLStream.write(space + "        </td>\n");
                resultHTMLStream.write(space + "        <td >" + keyArray[i][1] + "\n");
                resultHTMLStream.write(space + "        </td>\n");

                if (key == "pass2fail") {
                    resultHTMLStream.write(space + "        <td class='pass'>Pass\n");
                    resultHTMLStream.write(space + "        </td>\n");
                    resultHTMLStream.write(space + "        <td class='fail'>Fail\n");
                    resultHTMLStream.write(space + "        </td>\n");
                } else {
                    resultHTMLStream.write(space + "        <td class='fail'>Fail\n");
                    resultHTMLStream.write(space + "        </td>\n");
                    resultHTMLStream.write(space + "        <td class='pass'>Pass\n");
                    resultHTMLStream.write(space + "        </td>\n");
                }

                resultHTMLStream.write(space + "      </tr>\n");
            }
        }

        resultHTMLStream.write(space + "  </tbody>\n");
        resultHTMLStream.write(space + "</table><br /><br />\n");
    }

    var bodyContainerBoxTableTotal =  function(space) {
        resultHTMLStream.write(space + "<div class='box-table-all'>\n");
        resultHTMLStream.write(space + "  <table>\n");
        resultHTMLStream.write(space + "    <thead>\n");
        resultHTMLStream.write(space + "      <tr>\n");
        resultHTMLStream.write(space + "        <th rowspan='2'>Summary\n");
        resultHTMLStream.write(space + "        </th>\n");
        for (let i = 0; i < testPrefers.length; i++) {
            resultHTMLStream.write(space + "        <th colspan='2'>" + testPrefers[i] + "\n");
            resultHTMLStream.write(space + "        </th>\n");
        }

        resultHTMLStream.write(space + "      </tr>\n");
        resultHTMLStream.write(space + "      <tr>\n");
        for (let i = 0; i < testPrefers.length; i++) {
            resultHTMLStream.write(space + "        <th>Baseline\n");
            resultHTMLStream.write(space + "        </th>\n");
            resultHTMLStream.write(space + "        <th>Test Build\n");
            resultHTMLStream.write(space + "        </th>\n");
        }

        resultHTMLStream.write(space + "      </tr>\n");
        resultHTMLStream.write(space + "    </thead>\n");
        resultHTMLStream.write(space + "    <tbody>\n");

        let TableTotalDataArray = ["Total", "Pass", "Fail", "Block", "PassRate%"];
        for (let i = 0; i < TableTotalDataArray.length; i++) {
            resultHTMLStream.write(space + "      <tr>\n");
            resultHTMLStream.write(space + "        <th>" + TableTotalDataArray[i] + "\n");
            resultHTMLStream.write(space + "        </th>\n");

            for (let j = 0; j < testPrefers.length; j++) {
                resultHTMLStream.write(space + "        <td>" + pageDataTotal.get(testPrefers[j]).get("Baseline")[i] + "\n");
                resultHTMLStream.write(space + "        </td>\n");

                if (typeof pageDataTotal.get(testPrefers[j]).get("grasp")[i] == "undefined") {
                    resultHTMLStream.write(space + "        <td>N/A\n");
                } else {
                    resultHTMLStream.write(space + "        <td>" + pageDataTotal.get(testPrefers[j]).get("grasp")[i] + "\n");
                }

                resultHTMLStream.write(space + "        </td>\n");
            }

            resultHTMLStream.write(space + "      </tr>\n");
        }

        resultHTMLStream.write(space + "    </tbody>\n");
        resultHTMLStream.write(space + "  </table>\n");
        resultHTMLStream.write(space + "</div>\n");
    }

    var bodyContainerBoxTableLogPrefer = function(space, prefer) {
        resultHTMLStream.write(space + "<h3>Chromium log message for " + prefer + " prefer:</h3>\n");

        if (testPlatform == "Android") {
            resultHTMLStream.write(space + "<h3>NOTE: This is test case logs, not chromium runtime logs, because 'Permission denied'.</h3>\n");
        }

        resultHTMLStream.write(space + "<table class='box-table-log'><br />\n");
        resultHTMLStream.write(space + "  <tbody>\n");

        let logPath;
        if (os.type() == "Windows_NT") {
            logPath = debugPath + "\\debug-" + prefer + ".log";
        } else {
            logPath = debugPath + "/debug-" + prefer + ".log";
        }

        let fRead = fs.readFileSync(logPath);
        let fReadArray = fRead.toString().split("\n");

        for (let i = 1; i < fReadArray.length; i++) {
            resultHTMLStream.write(space + "    <tr>\n");
            resultHTMLStream.write(space + "      <td class='box-table-log-number'>" + i + "</td>\n");
            resultHTMLStream.write(space + "      <td class='box-table-log-text'>" + fReadArray[i] + "</td>\n");
            resultHTMLStream.write(space + "    </tr>\n");
        }

        resultHTMLStream.write(space + "  </tbody>\n");
        resultHTMLStream.write(space + "</table><br /><br />\n");
    }

    var bodyContainerBoxTable =  function(space) {
        resultHTMLStream.write(space + "<div class='box-table'>\n");
        resultHTMLStream.write(space + "  <div class='active' id='box-table-pass2fail'>\n");

        for (let i = 0; i < testPrefers.length; i++) {
            let flag = false;

            for (let j = 0; j < crashData.length; j++) {
                if (testPrefers[i] == crashData[j]) flag = true;
            }

            if (crashData.length !== 0 && flag) {
                continue;
            } else {
                bodyContainerBoxTablePrefer(space + "    ", testPrefers[i], "pass2fail");
            }
        }

        resultHTMLStream.write(space + "  </div>\n");
        resultHTMLStream.write(space + "  <div id='box-table-fail2pass'>\n");

        for (let i = 0; i < testPrefers.length; i++) {
            let flag = false;

            for (let j = 0; j < crashData.length; j++) {
                if (testPrefers[i] == crashData[j]) flag = true;
            }

            if (crashData.length !== 0 && flag) {
                continue;
            } else {
                bodyContainerBoxTablePrefer(space + "    ", testPrefers[i], "fail2pass");
            }
        }

        resultHTMLStream.write(space + "  </div>\n");

        for (let prefer of testPrefers) {
            resultHTMLStream.write(space + "  <div id='box-table-" + prefer + "'>\n");
            bodyContainerBoxTableLogPrefer(space + "    ", prefer);
            resultHTMLStream.write(space + "  </div>\n");
        }

        bodyContainerBoxTableTotal(space + "  ");
        bodyContainerNewTest(space + "  ");

        resultHTMLStream.write(space + "</div>\n");
    }

    var bodyContainerBox = function(space) {
        resultHTMLStream.write(space + "<h3>Result:</h3>\n");

        bodyContainerBoxMenu(space);
        bodyContainerBoxTable(space);
    }

    var bodyContainer = function(space) {
        resultHTMLStream.write(space + "<div class='container'>\n");

        bodyContainerVersion(space + "  ");
        bodyContainerCrash(space + "  ");
        bodyContainerSuggest(space + "  ");
        bodyContainerBox(space + "  ");

        resultHTMLStream.write(space + "</div>\n");
    }

    var createHtmlBody = function(space) {
        resultHTMLStream.write(space + "<body>\n");

        bodyContainer(space + "  ");

        resultHTMLStream.write(space + "</body>\n");
    }

    var createHtmlFile = function() {
        fs.writeFileSync(resultHTMLPath, "<!DOCTYPE html>\n");

        resultHTMLStream.write("<html>\n");

        createHtmlHead();
        createHtmlBody("  ");

        resultHTMLStream.write("</html>\n");
    }

    RClog("time", "mark");

    for (let prefer of testPrefers) {
        chromeOption = new Chrome.Options();
        testPrefer = prefer;
        graspDataSummary["total"] = 0;
        graspDataSummary["pass"] = 0;
        graspDataSummary["fail"] = 0;
        graspDataSummary["block"] = 0;
        continueFlag = false;
        remoteURL = "https://brucedai.github.io/webnnt/test/index-local.html";

        // Categories filter
        if (testPrefer === "macOS-Polyfill-Fast-WASM") {
            if (testPlatform === "Mac" && webmlPolyfill) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "macOS-Polyfill-Sustained-WebGL") {
            if (testPlatform === "Mac" && webmlPolyfill) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "macOS-WebNN-Fast-BNNS") {
            if (testPlatform === "Mac" && webnn) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "macOS-WebNN-Fast-MKLDNN") {
            if (testPlatform === "Mac" && webnn && supportSwitch) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--use-mkldnn")
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "macOS-WebNN-Sustained-MPS") {
            if (testPlatform === "Mac" && webnn) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Android-Polyfill-Fast-WASM") {
            if (testPlatform === "Android" && webmlPolyfill) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--disable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (testPrefer === "Android-Polyfill-Sustained-WebGL") {
            if (testPlatform === "Android" && webmlPolyfill) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--disable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (testPrefer === "Android-WebNN-Sustained-NNAPI") {
            if (testPlatform === "Android" && webnn) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--enable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (testPrefer === "Win-Polyfill-Fast-WASM") {
            if (testPlatform === "Windows" && webmlPolyfill) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML")
            } else {
                continue;
            }
        } else if (testPrefer === "Win-Polyfill-Sustained-WebGL") {
            if (testPlatform === "Windows" && webmlPolyfill) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML")
            } else {
                continue;
            }
        } else if (testPrefer === "Win-WebNN-Fast-MKLDNN") {
            if (testPlatform === "Windows" && webnn) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Win-WebNN-Sustained-DML") {
            if (testPlatform === "Windows" && webnn && supportSwitch) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--use-dml")
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Win-WebNN-Sustained-clDNN") {
            if (testPlatform === "Windows" && webnn) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Win-WebNN-Low-DML") {
            if (testPlatform === "Windows" && webnn && supportSwitch) {
                remoteURL = remoteURL + "?prefer=low";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--use-dml")
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-Polyfill-Fast-WASM") {
            if (testPlatform === "Linux" && webmlPolyfill) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-Polyfill-Sustained-WebGL") {
            if (testPlatform === "Linux" && webmlPolyfill) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Fast-MKLDNN") {
            if (testPlatform === "Linux" && webnn) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Sustained-clDNN") {
            if (testPlatform === "Linux" && webnn) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Fast-IE-MKLDNN") {
            if (testPlatform === "Linux" && webnn && supportSwitch) {
                remoteURL = remoteURL + "?prefer=fast";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--use-inference-engine")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Sustained-IE-clDNN") {
            if (testPlatform === "Linux" && webnn && supportSwitch) {
                remoteURL = remoteURL + "?prefer=sustained";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--use-inference-engine")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Low-IE-MYRIAD") {
            if (testPlatform === "Linux" && webnn && supportSwitch) {
                remoteURL = remoteURL + "?prefer=low";
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--use-inference-engine")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        }

        let logPath;
        if (testPlatform !== "Android") {
            if (os.type() == "Windows_NT") {
                logPath = process.cwd() + "\\output\\debug\\tmp";
                chromeOption = chromeOption.addArguments("--user-data-dir=" + logPath);

                if (!fs.existsSync(logPath)) {
                    fs.mkdirSync(logPath);
                }
            } else {
                logPath = process.cwd() + "/output/debug/debug-" + testPrefer + ".log";
                chromeOption = chromeOption.setChromeLogFile(logPath);
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

        // Wait mocha test finished
        await driver.wait(async function() {
            return driver.executeScript("return window.mochaFinish;").catch(function(err) {
                throw err;
            });
        }, 200000).then(async function() {
            RClog("console", "load remote URL is completed, no crash");

            // Output log file
            if (testPlatform == "Android") {
                let Path, sourceHTMLPath, logPath;
                if (os.type() == "Windows_NT") {
                    sourceHTMLPath = outputPath + "\\source-" + testPrefer + ".html";
                    Path = "file://" + process.cwd() + "\\output\\source-" + testPrefer + ".html";
                    logPath = process.cwd() + "\\output\\debug\\debug-" + testPrefer + ".log";
                } else {
                    sourceHTMLPath = outputPath + "/source-" + testPrefer + ".html";
                    Path = "file://" + process.cwd() + "/output/source-" + testPrefer + ".html";
                    logPath = process.cwd() + "/output/debug/debug-" + testPrefer + ".log";
                }

                await driver.executeScript("return document.documentElement.outerHTML").then(function(html) {
                    RClog("console", "dowload source html to " + sourceHTMLPath);

                    fs.createWriteStream(sourceHTMLPath, {flags: "w"}).write(html);
                });

                await driver.manage().logs().get("browser").then(function(Entrys) {
                    if (fs.existsSync(logPath)) {
                        fs.unlinkSync(logPath);
                    }

                    for (let entry of Entrys) {
                        fs.createWriteStream(logPath, {flags: "a"}).write(entry.message + "\n");
                    }

                    RClog("console", "dowload log file to " + logPath);
                });

                await driver.quit();
                await driver.sleep(2000);

                driver = new Builder()
                    .forBrowser("chrome")
                    .setChromeOptions(new Chrome.Options().setChromeBinaryPath(chromiumPath))
                    .build();

                RClog("time", "mark");

                await driver.get(Path);
            } else if (testPlatform == "Windows") {
                let readLogFile = process.cwd() + "\\output\\debug\\tmp\\chrome_debug.log";
                let writeLogFile = process.cwd() + "\\output\\debug\\debug-" + testPrefer + ".log";
                fs.writeFileSync(writeLogFile, fs.readFileSync(readLogFile));
            }
        }).catch(function(err) {
            RClog("debug", err);

            // Handler: page crashed -- 1
            if (err.message.search("session deleted because of page crash") != -1) {
                continueFlag = true;
                crashData.push(testPrefer);
                RClog("console", "remote URL is crashed");
            } else {
                throw err;
            }
        });

        RClog("time", "mark");

        // Handler: page crashed -- 2
        if (continueFlag) {
            await driver.sleep(2000);
            await driver.quit();
            await driver.sleep(2000);

            continue;
        }

        RClog("console", "checking with '" + testPrefer + "' prefer is start");
        RClog("console", "checking....");

        await graspResult();

        RClog("time", "mark");

        pageDataTotal.get(testPrefer).get("grasp").push(graspDataSummary["total"]);
        pageDataTotal.get(testPrefer).get("grasp").push(graspDataSummary["pass"]);
        pageDataTotal.get(testPrefer).get("grasp").push(graspDataSummary["fail"]);
        pageDataTotal.get(testPrefer).get("grasp").push(graspDataSummary["block"]);
        pageDataTotal.get(testPrefer).get("grasp").push(Math.round((graspDataSummary["pass"] / graspDataSummary["total"]) * 100).toString() + "%");

        await driver.sleep(2000);
        await driver.quit();
        await driver.sleep(2000);

        RClog("console", "checking with '" + testPrefer + "' prefer is completed");
    }

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

    await createHtmlFile();

    resultHTMLStream.end();

    driver = new Builder()
        .forBrowser("chrome")
        .setChromeOptions(new Chrome.Options().setChromeBinaryPath(chromiumPath))
        .build();

    RClog("time", "mark");

    await driver.get(resultHTMLPathFull);
})().then(function() {
    RClog("console", "checking chromium code is completed");
}).catch(function(err) {
    driver.quit();
    RClog("console", err);
});
