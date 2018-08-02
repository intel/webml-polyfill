| Op Type | WASM | WebGL2 | NNAPI | MPS | BNNS ([#8](https://github.com/intel/webml-polyfill/issues/8))
|----|------|--------|-------|-----|-----|
| ADD | yes | no ([#93](https://github.com/intel/webml-polyfill/issues/93)) | yes | no ([#92](https://github.com/intel/webml-polyfill/issues/92))| no
| AVERAGE_POOL_2D | yes | yes| yes | yes | yes
| CONCATENATION | yes | yes| yes | yes | no
| CONV_2D | yes | yes| yes | yes | yes
| DEPTHWISE_CONV_2D | yes | yes| yes | yes | no
| MAX_POOL_2D |  yes | yes| yes | yes | yes
| MUL |  yes | no ([#104](https://github.com/intel/webml-polyfill/issues/104)) | yes | no ([#117](https://github.com/intel/webml-polyfill/issues/117))| no
| RESHAPE |  yes | yes| yes | yes | no
| SOFTMAX |  yes | yes | yes | yes | yes
