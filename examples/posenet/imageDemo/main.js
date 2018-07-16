let util = new Utils();
let canvasSingle = document.getElementById('canvas');
let ctxSingle = canvasSingle.getContext('2d');
let canvasMulti = document.getElementById('canvas_2');
let ctxMulti = canvasMulti.getContext('2d');
async function DrawSingleandMulti(){
    let e = document.getElementById("backend");
    let backend = e.options[e.selectedIndex].text;
    switch(backend){
        case "WebGL":
            if (nnPolyfill.supportWebGL2){
                await util.init('WebGL2');
            }
            else{
                throw new Error("Do not support WebGL");
            }
            break;
        case "WASM":
            if (nnPolyfill.supportWasm){
                await util.init('WASM');
            }
            else{
                throw new Error("Do not support WASM");
            }
            break;
        case "WebML":
            if(nnNative){
                await util.init('WebML');
            }
            else{
                throw new Error("Do not support WebML");
            }
            break;
        default:
            break;
    }
    ctxSingle.clearRect(0, 0, canvasSingle.width, canvasSingle.height);
    ctxMulti.clearRect(0, 0, canvasMulti.width, canvasMulti.height);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxSingle);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxMulti);
    await util.predict(canvasSingle);
}

DrawSingleandMulti();

async function changeImage(){
    let _inputElement = document.getElementById('image').files[0];
    if(_inputElement!=undefined){
        ctxSingle.clearRect(0, 0, canvasSingle.width, canvasSingle.height);
        ctxMulti.clearRect(0, 0, canvasMulti.width, canvasMulti.height);
        let x = await getInput(_inputElement);
        await loadImage(x, ctxSingle);
        await loadImage(x, ctxMulti);
        util.predict(canvasSingle);
    }
}

