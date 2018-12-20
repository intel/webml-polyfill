// Reference https://github.com/Microsoft/onnxjs/blob/master/tools/build.ts

const child_process =  require('child_process');
const execSync = child_process.execSync;
const spawnSync = child_process.spawnSync;
const fs = require('fs');
const globby = require('globby');
const logger = require('npmlog');
const path = require('path');
const rimraf = require('rimraf');

logger.info('Build', 'Initializing...');

// Flags
// To trigger fetching some WASM dependencies and some post-processing
const buildWasm = process.argv.indexOf('--build-wasm') !== -1;
// To trigger a clean install
const cleanInstall = process.argv.indexOf('--clean-install') !== -1;

// Path variables
const ROOT = __dirname;
const DEPS = path.join(ROOT, 'deps');
const WASM_SRC = path.join(ROOT, 'src/nn/wasm/src');
const WASM_SRC_EXTERNAL = path.join(WASM_SRC, 'external');
const WASM_BUILD = path.join(WASM_SRC, 'build');
const WASM_TENSORFLOW = path.join(WASM_SRC, 'external/tensorflow');
const TENSORFLOW_DOWNLOADS = path.join(WASM_TENSORFLOW, 'tensorflow/contrib/makefile/downloads');
const TENSORFLOW_EIGEN = path.join(TENSORFLOW_DOWNLOADS, 'eigen');
const TENSORFLOW_GEMMLOWP = path.join(TENSORFLOW_DOWNLOADS, 'gemmlowp');
const BUILD_NN_OPS = path.join(WASM_BUILD, 'nn_ops.js');
const DEPS_EMSDK = path.join(DEPS, 'emsdk');
const DEPS_EMSDK_EMSCRIPTEN = path.join(DEPS_EMSDK, 'emscripten');
const EMSDK_BIN = path.join(DEPS_EMSDK, 'emsdk');
const NN_OPS = path.join(ROOT, 'src/nn/wasm/nn_ops.js');

logger.info('Build', 'Initialization completed. Start to build...');

logger.info('Build', 'Updating submodules...');
// Step 1: Clean folders if needed
logger.info('Build.SubModules', '(1/3) Cleaning dependencies folder...');
if (cleanInstall) {
  rimraf.sync(DEPS);
  rimraf.sync(WASM_SRC_EXTERNAL);
}
logger.info('Build.SubModules', `(1/3) Cleaning dependencies folder... ${cleanInstall ? 'DONE' : 'SKIPPED'}`);

// Step 2: Get dependencies (if needed)
logger.info('Build.SubModules', '(2/3) Fetching submodules...');
const update = spawnSync('git submodule update --init --recursive', {shell: true, stdio: 'inherit', cwd: ROOT});
if (update.status !== 0) {
  if (update.error) {
    console.error(update.error);
  }
  process.exit(update.status);
}
logger.info('Build.SubModules', '(2/3) Fetching submodules... DONE');

logger.info('Download tensorflow dependencies', '(3/3) Downloading... ');
if (!fs.existsSync(TENSORFLOW_EIGEN) || !fs.existsSync(TENSORFLOW_GEMMLOWP)) {
  const download = spawnSync('tensorflow/contrib/makefile/download_dependencies.sh', {shell: true, stdio: 'inherit', cwd: WASM_TENSORFLOW});
  if (download.status !== 0) {
    if (download.error) {
      console.error(download.error);
    }
    process.exit(download.status);
  }
}
logger.info('Download tensorflow dependencies', '(3/3) Downloading... DONE');

logger.info('Build', 'Updating submodules... DONE');

if (!fs.existsSync(WASM_BUILD)) {
  logger.info('Build', `Creating output folder: ${WASM_BUILD}`);
  fs.mkdirSync(WASM_BUILD);
}

logger.info('Build', 'Building WebAssembly sources...');
if (!buildWasm) {
  // if not building Wasm AND the file onnx-wasm.js is not present, create a place holder file
  if (!fs.existsSync(NN_OPS)) {
    logger.info('Build.Wasm', `Writing fallback target file: ${NN_OPS}`);
    fs.writeFileSync(NN_OPS, `;throw new Error("please build WebAssembly before use wasm backend.");`);
  }
} else {
  // Step 1: emsdk install (if needed)
  logger.info('Build.Wasm', '(1/4) Setting up emsdk...');
  if (!fs.existsSync(DEPS_EMSDK_EMSCRIPTEN)) {
    logger.info('Build.Wasm', 'Installing emsdk...');
    const install = spawnSync(`${EMSDK_BIN} install latest`, {shell: true, stdio: 'inherit', cwd: DEPS_EMSDK});
    if (install.status !== 0) {
      if (install.error) {
        console.error(install.error);
      }
      process.exit(install.status);
    }
    logger.info('Build.Wasm', 'Installing emsdk... DONE');

    logger.info('Build.Wasm', 'Activating emsdk...');
    const activate = spawnSync(`${EMSDK_BIN} activate latest`, {shell: true, stdio: 'inherit', cwd: DEPS_EMSDK});
    if (activate.status !== 0) {
      if (activate.error) {
        console.error(activate.error);
      }
      process.exit(activate.status);
    }
    logger.info('Build.Wasm', 'Activating emsdk... DONE');
  }
  logger.info('Build.Wasm', '(1/4) Setting up emsdk... DONE');

  // Step 2: Find path to TOOL_CHAIN
  logger.info('Build.Wasm', '(2/4) Find path to camke toolchain...');
  let TOOL_CHAIN = globby.sync('./emscripten/**/cmake/Modules/Platform/Emscripten.cmake', {cwd: DEPS_EMSDK})[0];
  console.log(TOOL_CHAIN);
  if (!TOOL_CHAIN) {
    logger.error('Build.Wasm', 'Unable to find camke toolchain. Try re-building with --clean-install flag.');
    process.exit(2);
  }
  TOOL_CHAIN = path.join(DEPS_EMSDK, TOOL_CHAIN);
  logger.info('Build.Wasm', `(2/4) Find path to TOOL_CHAIN... DONE, TOOL_CHAIN: ${TOOL_CHAIN}`);

  // Step 3: Set cmake tool-chain
  logger.info('Build.Wasm', '(3/4) Set cmake tool-chain...');
  // tslint:disable-next-line:non-literal-require
  const cmaketoolchain = execSync(`cmake -D CMAKE_TOOLCHAIN_FILE=${TOOL_CHAIN} ..`, {cwd: WASM_BUILD});
  
  if (cmaketoolchain.error) {
    console.error(cmaketoolchain.error);
    process.exit(cmaketoolchain.status);
  }

  logger.info('Build.Wasm', '(3/4) Preparing build config... DONE');

  // Step 4: Compile the source code to generate the Wasm file
  logger.info('Build.Wasm', '(4/4) Building...');

  const wasmBuild = spawnSync(`make`, {shell: true, stdio: 'inherit', cwd: WASM_BUILD});

  if (wasmBuild.error) {
    console.error(wasmBuild.error);
    process.exit(wasmBuild.status);
  }
  logger.info('Build.Wasm', '(4/4) Building... DONE');
  let move = spawnSync(`mv ${BUILD_NN_OPS} ${NN_OPS}`, {shell: true, stdio: 'inherit', cwd: WASM_BUILD});
}
logger.info('Build', `Building WebAssembly sources... ${buildWasm ? 'DONE' : 'SKIPPED'}`);