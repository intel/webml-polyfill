async function DrawSingleandMulti(){
    let util = new Utils();
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
    await util.predict();
}

DrawSingleandMulti();


