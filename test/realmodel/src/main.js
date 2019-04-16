const childProcess = require('child_process');
const path = require('path');

console.log('Begin Test...');

(async () => {
  let downloadModel = await childProcess.spawnSync(
    'node',
    [path.join(__dirname, 'downloadModelFile.js')],
    {stdio: [process.stdin, process.stdout, 'pipe']}
  );
  if (downloadModel.stderr.toString() != '') {
    console.log(downloadModel.stderr.toString());
    process.exit(1);
  }

  let unzipProcess = await childProcess.spawnSync(
    'node',
    [path.join(__dirname, 'unzipModel.js')],
    {stdio: [process.stdin, process.stdout, 'pipe']}
  );
  if (unzipProcess.stderr.toString() != '') {
    console.log(unzipProcess.stderr.toString());
    process.exit(1);
  }

  let getCaseOriginProcess = await childProcess.spawnSync(
    'node',
    [path.join(__dirname, 'getCaseOriginData.js')],
    {stdio: [process.stdin, process.stdout, 'pipe']}
  );
  if (getCaseOriginProcess.stderr.toString() != '') {
    console.log(getCaseOriginProcess.stderr.toString());
    process.exit(1);
  }

  let geterateCaseProcess = await childProcess.spawnSync(
    'node',
    [path.join(__dirname, 'generateCase.js')],
    {stdio: [process.stdin, process.stdout, 'pipe']}
  );
  if (geterateCaseProcess.stderr.toString() != '') {
    console.log(geterateCaseProcess.stderr.toString());
    process.exit(1);
  }

  let getHtmlProcess = await childProcess.spawnSync(
    'node',
    [path.join(__dirname, 'genHtml.js')],
    {stdio: [process.stdin, process.stdout, 'pipe']}
  );
  if (geterateCaseProcess.stderr.toString() != '') {
    console.log(geterateCaseProcess.stderr.toString());
    process.exit(1);
  }
})().then(() => {
  console.log('Test case generate finished');
}).catch((err) => {
  throw(err);
});
