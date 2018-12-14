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
var csvStream = csv.createWriteStream({headers: true}).transform(function(row) {return {
    "Feature": row.Feature,
    "Case Id": row.CaseId,
    "Test Case": row.TestCase,
    "BaseLine(Mac-MPS)": row.BLMMPS,
    "CheckResult(Mac-MPS)": row.CRMMPS,
    "BaseLine(Mac-BNNS)": row.BLMBNNS,
    "CheckResult(Mac-BNNS)": row.CRMBNNS,
    "BaseLine(Mac-WASM)": row.BLMWASM,
    "CheckResult(Mac-WASM)": row.CRMWASM,
    "BaseLine(Mac-WebGL)": row.BLMWebGL,
    "CheckResult(Mac-WebGL)": row.CRMWebGL,
    "BaseLine(Android-NNAPI)": row.BLANNAPI,
    "CheckResult(Android-NNAPI)": row.CRANNAPI,
    "BaseLine(Android-WASM)": row.BLAWASM,
    "CheckResult(Android-WASM)": row.CRAWASM,
    "BaseLine(Android-WebGL)": row.BLAWebGL,
    "CheckResult(Android-WebGL)": row.CRAWebGL,
    "BaseLine(Windows-clDNN)": row.BLWclDNN,
    "CheckResult(Windows-clDNN)": row.CRWclDNN,
    "BaseLine(Windows-WASM)": row.BLWWASM,
    "CheckResult(Windows-WASM)": row.CRWWASM,
    "BaseLine(Windows-WebGL)": row.BLWWebGL,
    "CheckResult(Windows-WebGL)": row.CRWWebGL,
    "BaseLine(Linux-clDNN)": row.BLLclDNN,
    "CheckResult(Linux-clDNN)": row.CRLclDNN,
    "BaseLine(Linux-WASM)": row.BLLWASM,
    "CheckResult(Linux-WASM)": row.CRLWASM,
    "BaseLine(Linux-WebGL)": row.BLLWebGL,
    "CheckResult(Linux-WebGL)": row.CRLWebGL
}});

var csvFilePath = outputPath + "/report-check-result.csv";
csvStream.pipe(fs.createWriteStream(csvFilePath));

var remoteURL, driver, backendModel, chromeOption, command, androidSN, adbPath, htmlPath;
var backendModels = [
    "Mac-WASM",
    "Mac-WebGL",
    "Mac-MPS",
    "Mac-BNNS",
    "Android-WASM",
    "Android-WebGL",
    "Android-NNAPI",
    "Windows-WASM",
    "Windows-WebGL",
    "Windows-clDNN",
    "Linux-WASM",
    "Linux-WebGL",
    "Linux-clDNN"
];

var RCjson = JSON.parse(fs.readFileSync("./config.json"));
var testPlatform = RCjson.platform;
var chromiumPath = RCjson.chromiumPath;

var baselinejson = JSON.parse(fs.readFileSync("./baseline/baseline.config.json"));
var versionChromium = baselinejson.Version.chromium;
var versionPolyfill = baselinejson.Version.polyfill;

/**
 * baseLineData = writeCSVData = {
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
var writeCSVData = new Map();
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
 * graspData = [
 *     total: value,
 *     pass: value,
 *     fail: value,
 *     block: value
 * ]
 */
var graspData = new Array();

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
}).on("end", function() {
    for (let key of baseLineData.keys()) {
        RClog("debug", "key: " + key);
    }
});

var continueFlag = false;
var debugFlag = false;
function RClog (target, message) {
    if (target == "console") {
        console.log("RC -- " + message);
    } else if (target == "debug") {
        if (debugFlag) console.log("RC -- " + message);
    } else {
        throw new Error("Not support target '" + target + "'");
    }
}

RClog("console", "checking runtime environment....");

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

var numberPasstoFail = 0;
var numberFailtoPass = 0;
var numberTotal = 0;

(async function() {
    RClog("console", "checking chromium code is start");

    var getInfo = async function(element, count, title, module) {
        return element.getAttribute("class").then(function(message) {
            let checkCaseStatus = null;
            if (message == "test pass pending") {
                checkCaseStatus = "N/A";
                graspData["block"] = graspData["block"] + 1
            } else if (message == "test pass fast" || message == "test pass slow" || message == "test pass medium") {
                checkCaseStatus = "Pass";
                graspData["pass"] = graspData["pass"] + 1
            } else if (message == "test fail") {
                checkCaseStatus = "Fail";
                graspData["fail"] = graspData["fail"] + 1
            } else {
                throw new Error("not support case status");
            }

            graspData["total"] = graspData["total"] + 1;
            module = module + "/" + count;

            RClog("debug", "'Feature': " + title);
            RClog("debug", "'Case Id': " + module);
            RClog("debug", "'Case Status': " + checkCaseStatus + "\n");

            let resultFlag = checkResult(backendModel, title, module, checkCaseStatus);
            if (resultFlag) {
                if (!writeCSVData.has(title + "-" + module)) {
                    writeCSVData.set(title + "-" + module, new Array());

                    writeCSVData.get(title + "-" + module)["Feature"] = title;
                    writeCSVData.get(title + "-" + module)["CaseId"] = module;
                    writeCSVData.get(title + "-" + module)["TestCase"] = baseLineData.get(title + "-" + module).get("TestCase");
                }

                let DataArray = writeCSVData.get(title + "-" + module);
                let baseLineStatus = baseLineData.get(title + "-" + module).get(backendModel);
                let name = baseLineData.get(title + "-" + module).get("TestCase");

                switch(backendModel) {
                    case "Mac-MPS":
                        DataArray["BLMMPS"] = baseLineStatus;
                        DataArray["CRMMPS"] = checkCaseStatus;
                        break;
                    case "Mac-BNNS":
                        DataArray["BLMBNNS"] = baseLineStatus;
                        DataArray["CRMBNNS"] = checkCaseStatus;
                        break;
                    case "Mac-WASM":
                        DataArray["BLMWASM"] = baseLineStatus;
                        DataArray["CRMWASM"] = checkCaseStatus;
                        break;
                    case "Mac-WebGL":
                        DataArray["BLMWebGL"] = baseLineStatus;
                        DataArray["CRMWebGL"] = checkCaseStatus;
                        break;
                    case "Android-NNAPI":
                        DataArray["BLANNAPI"] = baseLineStatus;
                        DataArray["CRANNAPI"] = checkCaseStatus;
                        break;
                    case "Android-WASM":
                        DataArray["BLAWASM"] = baseLineStatus;
                        DataArray["CRAWASM"] = checkCaseStatus;
                        break;
                    case "Android-WebGL":
                        DataArray["BLAWebGL"] = baseLineStatus;
                        DataArray["CRAWebGL"] = checkCaseStatus;
                        break;
                    case "Windows-clDNN":
                        DataArray["BLWclDNN"] = baseLineStatus;
                        DataArray["CRWclDNN"] = checkCaseStatus;
                        break;
                    case "Windows-WASM":
                        DataArray["BLWWASM"] = baseLineStatus;
                        DataArray["CRWWASM"] = checkCaseStatus;
                        break;
                    case "Windows-WebGL":
                        DataArray["BLWWebGL"] = baseLineStatus;
                        DataArray["CRWWebGL"] = checkCaseStatus;
                        break;
                    case "Linux-WASM":
                        DataArray["BLLWASM"] = baseLineStatus;
                        DataArray["CRLWASM"] = checkCaseStatus;
                        break;
                    case "Linux-WebGL":
                        DataArray["BLLWebGL"] = baseLineStatus;
                        DataArray["CRLWebGL"] = checkCaseStatus;
                        break;
                    case "Linux-clDNN":
                        DataArray["BLLclDNN"] = baseLineStatus;
                        DataArray["CRLclDNN"] = checkCaseStatus;
                        break;
                }

                if (baseLineStatus == "Pass" && checkCaseStatus == "Fail") {
                    pageData.get(backendModel).get("pass2fail").push([title, module + "-" + name]);
                } else {
                    pageData.get(backendModel).get("fail2pass").push([title, module + "-" + name]);
                }

                RClog("console", title + "-" + module + "-" + name);
                RClog("console", baseLineStatus + " : " + checkCaseStatus);
            }
        });
    }

    var graspResult = async function() {
        let actions = 0;
        let actionCount = 0;

        await driver.findElements(By.xpath("//ul[@id='mocha-report']/li[@class='suite']")).then(function(arrayTitles) {
            for (let i = 0; i < arrayTitles.length; i++) {
                arrayTitles[i].findElement(By.xpath("./h1/a")).getText().then(function(message) {
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

                                for (let k = 0; k < arrayCase.length; k++) {
                                    await getInfo(arrayCase[k], k + 1, title, module).then(function() {
                                        actions = actions + 1;
                                    });
                                }

                            });
                        } else {
                            for (let j = 0; j < arrayModules.length; j++) {
                                arrayModules[j].findElement(By.xpath("./h1/a")).getText().then(function(message) {
                                    let module = message.split("#")[1];

                                    arrayModules[j].findElements(By.xpath("./ul/li[@class='test pass fast' or " +
                                                                          "@class='test pass slow' or " +
                                                                          "@class='test fail' or " +
                                                                          "@class='test pass pending' or " +
                                                                          "@class='test pass medium']")).then(async function(arrayCase) {
                                        RClog("debug", "title: " + title + "    module: " + module + "    case: " + arrayCase.length);

                                        for (let k = 0; k < arrayCase.length; k++) {
                                            await getInfo(arrayCase[k], k + 1, title, module).then(function() {
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
                RClog("debug", baselinejson[backendModel]["total"] + " : " + actionCount);
            }

            return (actions == baselinejson[backendModel]["total"]);
        }, 500000).catch(function() {
            RClog("console", "total: " + baselinejson[backendModel]["total"] + " grasp: " + actionCount);
            throw new Error("failed to grasp all test result");
        });
    }

    var checkResult = function(backend, title, module, statusCheck) {
        let caseId = title + "-" + module;
        if (!baseLineData.has(caseId)) {
            throw new Error("no match test case: " + caseId);
        } else {
            if (statusCheck === baseLineData.get(caseId).get(backend)) {
                return false;
            } else {
                return true;
            }
        }
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
    }

    var createHtmlBodyContainerWarnning = function(space) {
        htmlStream.write(space + "<hr />\n");

        if (crashData.length !== 0) {
            htmlStream.write(space + "<div class='warnning' id='option_div'>\n");
            htmlStream.write(space + "  <h3>Warnning:</h3>\n");

            for (let i = 0; i < crashData.length; i++) {
                htmlStream.write(space + "  <p id='" + crashData[i] + "'>Crash happened when testing " +
                                 crashData[i] + ", please double check.</p>\n");
            }

            htmlStream.write(space + "</div>\n");
        }
    }

    var createHtmlBodyContainerSuggest = function(space) {
        for (let i = 0; i < testBackends.length; i++) {
            numberPasstoFail = numberPasstoFail + pageData.get(testBackends[i]).get("pass2fail").length;
            numberFailtoPass = numberFailtoPass + pageData.get(testBackends[i]).get("fail2pass").length;

            if (typeof pageDataTotal.get(testBackends[i]).get("grasp")[0] !== "undefined") {
                numberTotal = numberTotal + pageDataTotal.get(testBackends[i]).get("grasp")[0];
            }
        }

        htmlStream.write(space + "<div>\n");

        if (numberPasstoFail == 0 && crashData.length == 0) {
            htmlStream.write(space + "  <h3>PR Submission Proposal: <span class='suggest'>OK</span></h3>\n");
        } else {
            htmlStream.write(space + "  <h3>PR Submission Proposal: <span class='notsuggest'>Please improve your code</span></h3>\n");
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
            let pass2failArray = new Array();
            for (let key of baseLineData.keys()) {
                for (let i = 0; i < pageData.get(backend).get("pass2fail").length; i++) {
                    if ((key + "-" + baseLineData.get(key).get("TestCase")) ==
                        (pageData.get(backend).get("pass2fail")[i][0] + "-" + pageData.get(backend).get("pass2fail")[i][1])) {
                        pass2failArray.push([pageData.get(backend).get("pass2fail")[i][0], pageData.get(backend).get("pass2fail")[i][1]]);
                    }
                }
            }

            for (let i = 0; i < pass2failArray.length; i++) {
                htmlStream.write(space + "      <tr class='pass2fail'>\n");
                htmlStream.write(space + "        <td >" + pass2failArray[i][0] + "\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "        <td >" + pass2failArray[i][1] + "\n");
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
            let fail2passArray = new Array();
            for (let key of baseLineData.keys()) {
                for (let i = 0; i < pageData.get(backend).get("fail2pass").length; i++) {
                    if ((key + "-" + baseLineData.get(key).get("TestCase")) ==
                        (pageData.get(backend).get("fail2pass")[i][0] + "-" + pageData.get(backend).get("fail2pass")[i][1])){
                        fail2passArray.push([pageData.get(backend).get("fail2pass")[i][0], pageData.get(backend).get("fail2pass")[i][1]]);
                    }
                }
            }

            for (let i = 0; i < fail2passArray.length; i++) {
                htmlStream.write(space + "      <tr class='fail2pass'>\n");
                htmlStream.write(space + "        <td >" + fail2passArray[i][0] + "\n");
                htmlStream.write(space + "        </td>\n");
                htmlStream.write(space + "        <td >" + fail2passArray[i][1] + "\n");
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
        createHtmlBodyContainerWarnning(space + "  ");
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

    for (let i = 0; i < backendModels.length; i++) {
        chromeOption = new Chrome.Options();
        backendModel = backendModels[i];
        graspData["total"] = 0;
        graspData["pass"] = 0;
        graspData["fail"] = 0;
        graspData["block"] = 0;
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

        await driver.wait(async function() {
            driver.executeScript("return window.mochaFinish;").then(function(flag) {
                RClog("debug", flag);
            }).catch(function(err) {
                throw err;
            });

            return driver.executeScript("return window.mochaFinish;").catch(function(err) {throw err;});
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

        if (continueFlag) {
            await driver.sleep(2000);
            await driver.quit();
            await driver.sleep(2000);

            continue;
        }

        RClog("console", "checking with '" + backendModel + "' backend is start");

        RClog("console", "checking....");

        await graspResult();

        pageDataTotal.get(backendModel).get("grasp").push(graspData["total"]);
        pageDataTotal.get(backendModel).get("grasp").push(graspData["pass"]);
        pageDataTotal.get(backendModel).get("grasp").push(graspData["fail"]);
        pageDataTotal.get(backendModel).get("grasp").push(graspData["block"]);
        pageDataTotal.get(backendModel).get("grasp").push(Math.round((graspData["pass"] / graspData["total"]) * 100).toString() + "%");

        await driver.sleep(2000);
        await driver.quit();
        await driver.sleep(2000);

        RClog("console", "checking with '" + backendModel + "' backend is completed");
    }

    for (let value of writeCSVData.values()) {
        csvStream.write(value);
    }

    await createHtmlFile();

    htmlStream.end();
    csvStream.end();

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

    await driver.get(htmlPath);
})().then(function() {
    RClog("console", "checking chromium code is completed");
}).catch(function(err) {
    driver.quit();
    RClog("console", err);
});
