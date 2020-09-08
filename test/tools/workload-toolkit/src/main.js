const childProcess = require("child_process");

const args = process.argv.slice(2);

(async () => {
  if (args.length === 0) {
    console.log("===== Start workload testing... ======");
  } else {
    console.log("===== Start Regression testing... ======");
  }

  if (args.length === 0 || (args.length === 1 && args[0] === 'without-baseline')) {
    let processDownload = await childProcess.spawnSync(
      "node",
      ["./util/download-build.js"],
      {stdio: [process.stdin, process.stdout, "pipe"]}
    );
    if (processDownload.stderr.toString() != "") {
      console.log(processDownload.stderr.toString());
      process.exit(1);
    }
    let processInstall = await childProcess.spawnSync(
      "node",
      ["./util/install-build.js"],
      {stdio: [process.stdin, process.stdout, "pipe"]}
    );
    if (processInstall.stderr.toString() != "") {
      console.log(processInstall.stderr.toString());
      process.exit(1);
    }
    let processRunTest = await childProcess.spawnSync(
      "node",
      ["./util/run-test.js"],
      {stdio: [process.stdin, process.stdout, "pipe"]}
    );
    if (processRunTest.stderr.toString() != "") {
      console.log(processRunTest.stderr.toString());
      process.exit(1);
    }
  }

  if (args.length === 1) {
    // re-run test for egression check
    let RunRegressionTestInstall = await childProcess.spawnSync(
      "node",
      ["./util/install-build.js", "regression-test"],
      {stdio: [process.stdin, process.stdout, "pipe"]}
    );
    if (RunRegressionTestInstall.stderr.toString() != "") {
      console.log(RunRegressionTestInstall.stderr.toString());
      process.exit(1);
    }
    let processRunRegressionTest = await childProcess.spawnSync(
      "node",
      ["./util/run-test.js", "regression-test"],
      {stdio: [process.stdin, process.stdout, "pipe"]}
    );
    if (processRunRegressionTest.stderr.toString() != "") {
      console.log(processRunRegressionTest.stderr.toString());
      process.exit(1);
    }
    let processResult = await childProcess.spawnSync(
      "node",
      ["./util/compare-result.js"],
      {stdio: [process.stdin, process.stdout, "pipe"]}
    );
    if (processResult.stderr.toString() != "") {
      console.log(processResult.stderr.toString());
      process.exit(1);
    }
    console.log("========== Completed Regression testing! =======");
  } else {
    console.log("========== Completed workload testing! =======");
  }
})();
