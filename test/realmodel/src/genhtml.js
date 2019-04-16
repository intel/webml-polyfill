require('../lib/jsonOperation.js');
const fs = require('fs');
const path = require('path');

let buf = '';
let caseList = '\n';
let htmlValue

let filePath = path.join(__dirname, '..', 'testcase', `${JSON_DATA.getModelName()}.txt`);
if (!fs.existsSync(filePath)) throw (`Can't get ${filePath}`);
let stream = fs.createReadStream(filePath, { flags: 'r', encoding: 'utf-8' });
stream.on('data', function (d) {
  buf += d.toString();
});
stream.on('end', () => {
  buf = JSON.parse(buf);
  generateHtml(buf)
  htmlValue = begin + caseList + end;
  saveHtml(htmlValue, `${JSON_DATA.getModelName()}.html`);
});

let begin = `
<html>
<head>
  <meta charset="utf-8">
  <title>WebML Polyfill | Mocha Tests</title>
  <link href='./static/mocha/3.0.2/mocha.css' rel='stylesheet'>
</head>
<body>
  <div id="mocha"></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mocha/4.0.1/mocha.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/chai/4.1.2/chai.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/fetch/2.0.3/fetch.min.js"></script>
  <script src="../dist/webml-polyfill.js"></script>
  <script src="./utils.js"></script>
  <script>
    window.onload = setOptions;
  </script>
  <script>
    mocha.setup('bdd');
  </script>`


 let end = ` <script>
    mocha.globals(['jQuery']);
    mocha.run();
  </script>
</body>
</html>
`
async function saveHtml(input, output) {
  let saveFileDirs = path.join(__dirname, '..', '..');
  let saveStream = fs.createWriteStream(path.join(saveFileDirs, output), { flags: 'w', encoding: 'utf-8' });
  saveStream.on('error', (err) => {
    console.error(err);
  });
  if (typeof (input) === 'object') {
    saveStream.write(JSON.stringify(input));
  } else {
    saveStream.write(input);
  }
  saveStream.end();
}
async function generateHtml(data) {
  for (let i = 0; i < data.length; i++) {
    let str = ` <script src="./realmodel/testcase/${data[i]}"></script>\n`
    caseList += str;
  }
}