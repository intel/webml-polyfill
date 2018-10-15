| Op Type | WASM | WebGL2 | NNAPI | MPS | BNNS ([#8](https://github.com/intel/webml-polyfill/issues/8)) | clDNN |
|----|------|--------|-------|-----|-----|-----|
| ADD | yes | yes | yes | no ([#92](https://github.com/intel/webml-polyfill/issues/92))| no | yes |
| AVERAGE_POOL_2D | yes | yes| yes | yes | yes | yes |
| CONCATENATION | yes | yes| yes | yes | yes | yes |
| CONV_2D | yes | yes| yes | yes | yes | yes |
| DEPTHWISE_CONV_2D | yes | yes| yes | yes | no | yes |
| MAX_POOL_2D |  yes | yes| yes | yes | yes | yes |
| MUL |  yes | yes | yes | no ([#117](https://github.com/intel/webml-polyfill/issues/117))| no | yes |
| RESHAPE |  yes | yes| yes | yes | no | yes |
| SOFTMAX |  yes | yes | yes | yes | yes | yes |
