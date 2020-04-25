// To copy tfjs-backend-wasm.wasm file

const fs = require('fs');
const logger = require('npmlog');

const clearWasm = process.argv.indexOf('--clear-wasm') !== -1;

const originWasmPath = './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm';
const originWasm = fs.readFileSync(originWasmPath);

const copyPath = [];
copyPath.push('./examples/emotion_analysis/tfjs-backend-wasm.wasm');
copyPath.push('./examples/face_recognition/tfjs-backend-wasm.wasm');
copyPath.push('./examples/facial_landmark_detection/tfjs-backend-wasm.wasm');
copyPath.push('./examples/image_classification/tfjs-backend-wasm.wasm');
copyPath.push('./examples/object_detection/tfjs-backend-wasm.wasm');
copyPath.push('./examples/semantic_segmentation/tfjs-backend-wasm.wasm');
copyPath.push('./examples/simple/tfjs-backend-wasm.wasm');
copyPath.push('./examples/skeleton_detection/tfjs-backend-wasm.wasm');
copyPath.push('./examples/speech_commands/tfjs-backend-wasm.wasm');
copyPath.push('./examples/super_resolution/tfjs-backend-wasm.wasm');
copyPath.push('./test/tfjs-backend-wasm.wasm');
copyPath.push('./workload/tfjs-backend-wasm.wasm');

if (clearWasm) {
    for (i in copyPath) {
        if (fs.existsSync(copyPath[i])) {
            fs.unlinkSync(copyPath[i]);
        }
    }
    logger.info('Clear the previous tfjs-backend-wasm.wasm file.');
}

for (i in copyPath) {
    if (!fs.existsSync(copyPath[i])) {
        fs.writeFileSync(copyPath[i], originWasm);
    }
}
logger.info('tfjs-backend-wasm.wasm file is ready.');