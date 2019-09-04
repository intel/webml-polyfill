const Builder = require("../node_modules/selenium-webdriver").Builder;
const By = require("../node_modules/selenium-webdriver").By;
const Until = require("../node_modules/selenium-webdriver").until;
const Chrome = require("../node_modules/selenium-webdriver/chrome");
const csv = require("../node_modules/fast-csv");
const execSync = require("child_process").execSync;
const cheerio = require("cheerio");
const fs = require("fs");
const os = require("os");
require("chromedriver");

var debugFlag = false;
var timeFlag = false;
function RClog (target, message) {
    if (target == "console") {
        console.log("PLA -- " + message);
    } else if (target == "debug") {
        if (debugFlag) console.log("PLA -- " + message);
    } else if (target == "time") {
        // For performance testing
        if (timeFlag) {
            console.log("PLA -- running time: " + process.uptime() + "s");
        }
    } else {
        throw new Error("Not support target '" + target + "'");
    }
}

RClog("console", "checking config.json file....");

function jsonTypeCheck (json, field, expectType) {
    if (typeof json[field] == expectType) {
        return true;
    } else {
        throw new Error("Type of 'JSON." + field + "' is not '" + expectType + "'");
    }
}

var testPlatform, chromiumPath, supportSwitch, API, preference ;
var RCjson = JSON.parse(fs.readFileSync("./config.json"));
if (jsonTypeCheck(RCjson, "platform", "string")) {
    testPlatform = RCjson.platform;
}

if (jsonTypeCheck(RCjson, "chromiumPath", "string")) {
    chromiumPath = RCjson.chromiumPath;
}

if (jsonTypeCheck(RCjson, "supportSwitch", "boolean")) {
    supportSwitch = RCjson.supportSwitch;
}

if (jsonTypeCheck(RCjson, "API", "string")) {
    API = RCjson.API;
}

if (jsonTypeCheck(RCjson, "preference", "string")) {
    preference = RCjson.preference;
}

var getLDLibraryPath = function() {
    // The path of chromium is '.../chromium-mac/Chromium.app/Contents/MacOS/Chromium'.
    // And the path of LD library is '.../chromium-mac/Chromium.app/Contents/Versions/75.0.3739.0/Chromium Framework.framework/Libraries/'.
    // To get the path of LD library from the path of chromium, we need to tailor string of chromium path.
    // And the length of 'MacOS/Chromium' is 14 that must be deleted.
    let basePath = chromiumPath.slice(0, -14) + "Versions/";
    let fileNames = fs.readdirSync(basePath);
    for (let fileName of fileNames) {
        if (fs.statSync(basePath + fileName).isDirectory()) {
            basePath = basePath + fileName + "/Chromium Framework.framework/Libraries/";
            break;
        }
    }
    return basePath;
}

var testPrefers = new Array();
if (testPlatform == "Linux") {
    if (preference === "low") {
        testPrefers.push("Linux-WebNN-Low-IE-MYRIAD");
    } else {
        if (API === "polyfill") {
            if (preference === "fast") {
                testPrefers.push("Linux-Polyfill-Fast-WASM");
            } else if (preference === "sustained") {
                testPrefers.push("Linux-Polyfill-Sustained-WebGL");
            }
        }

        if (API === "webnn") {
            if (preference === "fast") {
                testPrefers.push("Linux-WebNN-Fast-MKLDNN");
            } else if (preference === "sustained") {
                testPrefers.push("Linux-WebNN-Sustained-clDNN");
            }

            if (supportSwitch) {
                testPrefers.push("Linux-WebNN-Fast-IE-MKLDNN");
                testPrefers.push("Linux-WebNN-Sustained-IE-clDNN");
            }
        }
    }
} else if (testPlatform == "Android") {
    if (API === "polyfill") {
        if (preference === "fast") {
            testPrefers.push("Android-Polyfill-Fast-WASM");
        } else if (preference === "sustained") {
            testPrefers.push("Android-Polyfill-Sustained-WebGL");
        }    
    }

    if (API === "webnn") {
        testPrefers.push("Android-WebNN-Sustained-NNAPI");
    }
} else if (testPlatform == "Mac") {
    if (API === "polyfill") {
        if (preference === "fast") {
            testPrefers.push("macOS-Polyfill-Fast-WASM");
        } else if (preference === "sustained") {
            testPrefers.push("macOS-Polyfill-Sustained-WebGL");
        }
    }

    if (API === "webnn") {
        if (preference === "fast") {
            testPrefers.push("macOS-WebNN-Fast-BNNS");
        } else if (preference === "sustained") {
            testPrefers.push("macOS-WebNN-Sustained-MPS");
        }

        if (supportSwitch) {
            // Add process ENV
            process.env.LD_LIBRARY_PATH = getLDLibraryPath();
            testPrefers.push("macOS-WebNN-Fast-MKLDNN");
        }
    }
} else if (testPlatform == "Windows") {
    if (API === "polyfill") {
        if (preference === "fast") {
            testPrefers.push("Win-Polyfill-Fast-WASM");
        } else if (preference === "sustained") {
            testPrefers.push("Win-Polyfill-Sustained-WebGL");
        }  
    }

    if (API === "webnn") {
        if (preference === "fast") {
            testPrefers.push("Win-WebNN-Fast-MKLDNN");
        } else if (preference === "sustained") {
            testPrefers.push("Win-WebNN-Sustained-clDNN");
        }

        if (supportSwitch) {
            testPrefers.push("Win-WebNN-Sustained-DML");
        }
    }
}

RClog("console", "prefers: " + testPrefers);

RClog("console", "checking runtime environment....");

var command, androidSN, adbPath;
if (testPlatform == "Android") {
    RClog("console", "runtime environment: android");

    var sys = os.type();

    if (sys == "Linux") {
        adbPath = "../lib/adb-tool/Linux/adb";

        try {
            command = "killall adb";
            execSync(command, {encoding: "UTF-8", stdio: "pipe"});
        } catch(e) {
            if (e.message.search("no process found") == -1) {
                throw e;
            }
        }
    } else if (sys == "Darwin") {
        adbPath = "../lib/adb-tool/Mac/adb";

        try {
            command = "killall adb";
            execSync(command, {encoding: "UTF-8", stdio: "pipe"});
        } catch(e) {
            if (e.message.search("No matching processes") == -1) {
                throw e;
            }
        }
    } else if (sys == "Windows_NT") {
        adbPath = "..\\lib\\adb-tool\\Windows\\adb";

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

(async function() {
    RClog("console", "checking chromium code is start");

    for (let prefer of testPrefers) {
        chromeOption = new Chrome.Options();
        testPrefer = prefer;
        continueFlag = false;
        var modelName = RCjson.modelName;
        var urlServer = RCjson.urlServer
        if (modelName.length == 2) {
            if (modelName[0] === "squeezenet1.1") {
                remoteURL = `http://${urlServer}/webml-polyfill/test/real_squeezenet1.1_mobilenetv2-1.0.html`;
            } else if (modelName[0] === "mobilenetv2-1.0") {
                remoteURL = `http://${urlServer}/webml-polyfill/test/real_mobilenetv2-1.0_squeezenet1.1.html`;
            }
        }else if (modelName.length == 1) {
            if (modelName[0] === "squeezenet1.1") {
                remoteURL = `http://${urlServer}/webml-polyfill/test/real_squeezenet1.1.html`;
            } else if (modelName[0] === "mobilenetv2-1.0") {
                remoteURL = `http://${urlServer}/webml-polyfill/test/real_mobilenetv2-1.0.html`;
            }
        }

        // Categories filter
        if (testPrefer === "macOS-Polyfill-Fast-WASM") {
            if (testPlatform === "Mac" && API === "polyfill") {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "macOS-Polyfill-Sustained-WebGL") {
            if (testPlatform === "Mac" && API === "polyfill") {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "macOS-WebNN-Fast-BNNS") {
            if (testPlatform === "Mac" && API === "webnn") {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "macOS-WebNN-Fast-MKLDNN") {
            if (testPlatform === "Mac" && API === "webnn" && supportSwitch) {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--use-mkldnn")
                    .addArguments("--no-sandbox")
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "macOS-WebNN-Sustained-MPS") {
            if (testPlatform === "Mac" && API === "webnn") {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Android-Polyfill-Fast-WASM") {
            if (testPlatform === "Android" && API === "polyfill") {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--disable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (testPrefer === "Android-Polyfill-Sustained-WebGL") {
            if (testPlatform === "Android" && API === "polyfill") {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--disable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (testPrefer === "Android-WebNN-Sustained-NNAPI") {
            if (testPlatform === "Android" && API === "webnn") {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .androidPackage("org.chromium.chrome")
                    .addArguments("--enable-features=WebML")
                    .androidDeviceSerial(androidSN);
            } else {
                continue;
            }
        } else if (testPrefer === "Win-Polyfill-Fast-WASM") {
            if (testPlatform === "Windows" && API === "polyfill") {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML")
            } else {
                continue;
            }
        } else if (testPrefer === "Win-Polyfill-Sustained-WebGL") {
            if (testPlatform === "Windows" && API === "polyfill") {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML")
            } else {
                continue;
            }
        } else if (testPrefer === "Win-WebNN-Fast-MKLDNN") {
            if (testPlatform === "Windows" && API === "webnn") {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Win-WebNN-Sustained-DML") {
            if (testPlatform === "Windows" && API === "webnn" && supportSwitch) {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--use-dml")
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Win-WebNN-Sustained-clDNN") {
            if (testPlatform === "Windows" && API === "webnn") {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Win-WebNN-Low-DML") {
            if (testPlatform === "Windows" && API === "webnn" && supportSwitch) {
                remoteURL = remoteURL + `?prefer=low&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--use-dml")
                    .addArguments("--enable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-Polyfill-Fast-WASM") {
            if (testPlatform === "Linux" && API === "polyfill") {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-Polyfill-Sustained-WebGL") {
            if (testPlatform === "Linux" && API === "polyfill") {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--disable-features=WebML");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Fast-MKLDNN") {
            if (testPlatform === "Linux" && API === "webnn") {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Sustained-clDNN") {
            if (testPlatform === "Linux" && API === "webnn") {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Fast-IE-MKLDNN") {
            if (testPlatform === "Linux" && API === "webnn" && supportSwitch) {
                remoteURL = remoteURL + `?prefer=fast&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--use-inference-engine")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Sustained-IE-clDNN") {
            if (testPlatform === "Linux" && API === "webnn" && supportSwitch) {
                remoteURL = remoteURL + `?prefer=sustained&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--use-inference-engine")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        } else if (testPrefer === "Linux-WebNN-Low-IE-MYRIAD") {
            if (testPlatform === "Linux" && API === "webnn" && supportSwitch) {
                remoteURL = remoteURL + `?prefer=low&iterations=${RCjson.iterations}&API=${API}&platform=${testPlatform}&supportSwitch=${supportSwitch}`;
                chromeOption = chromeOption
                    .setChromeBinaryPath(chromiumPath)
                    .addArguments("--enable-features=WebML")
                    .addArguments("--use-inference-engine")
                    .addArguments("--no-sandbox");
            } else {
                continue;
            }
        }

        driver = new Builder()
            .forBrowser("chrome")
            .setChromeOptions(chromeOption)
            .build();

        await driver.get(remoteURL);
        await driver.wait(Until.elementLocated(By.xpath("//*[@id='mocha-stats']/li[1]/canvas")), 200000).then(function() {
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
        },5000000000).then(async function() {
            RClog("console", "load remote URL is completed, no crash");
        }).catch(function(err) {
            RClog("debug", err);

            // Handler: page crashed -- 1
            if (err.message.search("session deleted because of page crash") != -1) {
                continueFlag = true;
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
        RClog("console", "checking....\n");

        RClog("time", "mark");

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

    driver = new Builder()
        .forBrowser("chrome")
        .setChromeOptions(new Chrome.Options().setChromeBinaryPath(chromiumPath))
        .build();

    RClog("time", "mark");

})().then(function() {
    RClog("console", "checking chromium code is completed");
}).catch(function(err) {
    driver.quit();
    RClog("console", err);
});