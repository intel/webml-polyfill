# System Memory
Get physical memory information via `chrome.system.memory.getInfo()` method, and it gathers memory information once per second (in MB). 

## Install the extension
1. Get codes
 ```sh
 $ git clone https://github.com/intel/webml-polyfill.git
 ```

2. Open Chrome and visit `chrome://extensions`, enable `Developer Mode` by clicking the toggle switch next to `Developer mode`

3. Click the `Load unpacked` button and select the extension directory `webml-polyfill/test/tools/SystemMemory` (the extension has been successfully installed, you can see the extension icon in the Chrome toolbar, to the right of the address bar)

## Test
1. Open the test page (Better to clear other tasks to get the accurate data)

2. Click the extension icon (the script will be running in the background)

3. Start the testing

4. When the test is complete, press `Ctrl+Shift+Y` to save and download the memory data file `memory.txt`. Press `Command+Shift+Y` instead on macOS platform.
