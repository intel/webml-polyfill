const fs = require('fs');
const path = require('path');

let buf = '';
let caseList = '\n';
let htmlValue;
var RCjson = JSON.parse(fs.readFileSync("./config.json"));
var modelName = RCjson.modelName;
let result = 'real';
for (i = 0; i < modelName.length; i++) {
  let filePath1 = path.join(__dirname, '..', '..', '..', 'realmodel', 'testcase', `${modelName[i]}`, `${modelName[i]}.txt`);
  if (!fs.existsSync(filePath1)) throw (`Can't get ${filePath1}`);
  let data_model = fs.readFileSync(filePath1);
  data_model_length = JSON.parse(data_model);
  generateHtml(data_model_length, i);
  result += `_${modelName[i]}`;
};
result += '.html';
let filePath = path.join(__dirname, '..', '..', '..', 'realmodel', 'testcase', `${modelName[0]}`, `${modelName[0]}.txt`);
let stream = fs.createReadStream(filePath, {flags: 'r', encoding: 'utf-8'});
stream.on('data', function (d) {
  buf += d.toString();
});
stream.on('end', () => {
  htmlValue = begin + caseList + end;
  saveHtml(htmlValue, result);
});

let begin = `
<html>
<head>
  <style type="text/css">
  table {
    border: 1px solid black;
    border-collapse: collapse;
  }
  td,th {
    border: 1px solid black;
  }
  </style>
  <meta charset="utf-8">
  <title>WebML Polyfill | Mocha Tests</title>
  <link href='./static/mocha/3.0.2/mocha.css' rel='stylesheet'>
</head>
<body>
  <p id="avg" style="display:none"></p>
  <div id="result" style="display:none">
    <p id="testPlatform"></p>
    <p id="iterations">Iterations: 1(warming up) + </p>
    <table>
      <tr>
      <th>Layer</th>
      <th>Model_name</th>
      <th>Ops</th>
      <th>Avg_time(ms)</th>
      <th>Bias</th>
      <th>Weight</th>
      <th>Input_dimensions</th>
      <th>Output_dimensions</th>
      <th>Stride</th>
      <th>Filter</th>
      <th>Padding</th>
      <th>Activation</th>
      <th>Axis</th>
      <th>ShapeLen</th>
      <th>ShapeValues</th>
      </tr>
    </table>
  </div>
  <div id="mocha"></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mocha/4.0.1/mocha.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/chai/4.1.2/chai.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/fetch/2.0.3/fetch.min.js"></script>
  <script src="../dist/webml-polyfill.js"></script>
  <script src="./utils.js"></script>
  <script>
    setOptionsPerLayer();
  </script>
  <script>
    mocha.setup('bdd');
    mocha.setup({timeout:50000});
  </script>`


let end = ` <script>
  mocha.globals(['jQuery']);
  mocha.run(async function() {
    var obj = document.getElementById("avg");
    var avg_text = obj.innerHTML;
    avg_text = '[' + avg_text.slice(0, -1) + ']';
    avg_data = JSON.parse(avg_text);
    setTimeout(() => {
      var table = document.getElementsByTagName('table')[0];
      for (var i = 0; i < avg_data.length; i++) {
        var tr = table.insertRow(table.rows.length);
        var obj = avg_data[i];
        for (var p in obj) {
          if (p == null) {
            throw new Error('There is no content here');
          }
          var td = tr.insertCell(tr.cells.length);
          td.innerText = obj[p];
        }
      };
    }, 300);
    var reg = new RegExp("(^|&)prefer=([^&]*)&iterations=([^&]*)&API=([^&]*)&platform=([^&]*)&supportSwitch=([^&]*)(&|$)");
    var r = window.location.search.substr(1).match(reg);
    if (r != null) {
      var prefer = unescape(r[2]).toLowerCase();
      var iterations = unescape(r[3]).toLowerCase();
      var API = unescape(r[4]).toLowerCase();
      var testPlatform = unescape(r[5]);
      var supportSwitch = unescape(r[6]).toLowerCase();
    };
    document.getElementById("iterations").insertAdjacentText("beforeend", iterations);
    if (supportSwitch === "true") {
      if (testPlatform === "Mac" && API === "webnn" && prefer === "fast") {
        document.getElementById("testPlatform").insertAdjacentText("beforeend", testPlatform + ' / ' + API + ' / ' + prefer + ' / ' + 'MKLDNN');
      } else if (testPlatform === "Windows" && API === "webnn" && prefer === "sustained") {
        document.getElementById("testPlatform").insertAdjacentText("beforeend", testPlatform + ' / ' + API + ' / ' + prefer + ' / ' + 'DML');
      } else if (testPlatform === "Windows" && API === "webnn" && prefer === "low") {
        document.getElementById("testPlatform").insertAdjacentText("beforeend", testPlatform + ' / ' + API + ' / ' + prefer + ' / ' + 'DML');
      } else if (testPlatform === "Linux" && API === "webnn" && prefer === "fast") {
        document.getElementById("testPlatform").insertAdjacentText("beforeend", testPlatform + ' / ' + API + ' / ' + prefer + ' / ' + 'IE-MKLDNN');
      } else if (testPlatform === "Linux" && API === "webnn" && prefer === "sustained") {
        document.getElementById("testPlatform").insertAdjacentText("beforeend", testPlatform + ' / ' + API + ' / ' + prefer + ' / ' + 'IE-clDNN');
      } else if (testPlatform === "Linux" && API === "webnn" && prefer === "low") {
        document.getElementById("testPlatform").insertAdjacentText("beforeend", testPlatform + ' / ' + API + ' / ' + prefer + ' / ' + 'IE-MYRIAD');
      } else {
        document.getElementById("testPlatform").insertAdjacentText("beforeend", testPlatform + ' / ' + API + ' / ' + prefer);
      }
    } else {
      document.getElementById("testPlatform").insertAdjacentText("beforeend", testPlatform + ' / ' + API + ' / ' + prefer);
    }
    document.getElementById("result").style.display = "block";
  });
  </script>
</body>
</html>
`
async function saveHtml(input, output) {
  let saveFileDirs = path.join(__dirname, '..', '..', '..');
  let saveStream = fs.createWriteStream(path.join(saveFileDirs, output), {flags: 'w', encoding: 'utf-8'});
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

async function generateHtml(data, number) {
  for (let i = 0; i < data.length; i++) {
    let str = ` <script src="./realmodel/testcase/${modelName[number]}/${data[i]}"></script>\n`;
    caseList += str;
  }
}
