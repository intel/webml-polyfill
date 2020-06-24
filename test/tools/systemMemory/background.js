let Data = [];
let timer;

function bytesToMegaBytes(number) {
  return (number / 1024 / 1024).toFixed(6);
}

function saveAndDownloadFile(Data) {
    let filename = "memory.txt";
    let blob = new Blob([Data], { type: 'text/json' }),
        e = document.createEvent('MouseEvents'),
        a = document.createElement('a');
    a.download = filename;
    a.href = window.URL.createObjectURL(blob);
    a.dataset.downloadurl = ['text/json', a.download, a.href];
    e.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
    a.dispatchEvent(e);
    console.log("Save memery to " + filename);
  } 
  
function stoper() {
  clearTimeout(timer);
  Data = [];
  console.log("Clear timeout");
}

function init() {
  let memoryInUse = 0;
  // Get memory information
  (function getMemoryInfo() {
    chrome.system.memory.getInfo(function(memory) {
      memoryInUse = memory.capacity - memory.availableCapacity;
      Data.push(bytesToMegaBytes(memoryInUse));
    });
    timer = setTimeout(getMemoryInfo, 1000);
  })();
}

chrome.browserAction.onClicked.addListener(function(tab){
  init();
})

chrome.commands.onCommand.addListener(function(command){
  console.log('Command: ', command);
  saveAndDownloadFile(Data);
  stoper();
})
